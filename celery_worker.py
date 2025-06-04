# celery_worker.py

import os
# Elimina la importación de sqlite3, ya no se usará
# import sqlite3 
from datetime import datetime, date
from celery import Celery
from dotenv import load_dotenv
import time
import requests
import cv2
# Quita la importación de mediapipe si no lo usas activamente o no está en requirements.txt del worker
# import mediapipe as mp 
import subprocess
import json
import secrets
from evaluator import evaluate_interaction # ¡Asegúrate de que esta línea esté presente!

import boto3
from botocore.exceptions import ClientError

# --- Nuevas importaciones para PostgreSQL ---
import psycopg2
from urllib.parse import urlparse

load_dotenv()

# --- Configuración de rutas para archivos temporales (¡Ahora /tmp para volátiles!) ---
# /tmp es el lugar estándar para archivos temporales en Linux/Docker, que son efímeros.
TEMP_PROCESSING_FOLDER = os.getenv("TEMP_PROCESSING_FOLDER", "/tmp/leo_trainer_processing") 
os.makedirs(TEMP_PROCESSING_FOLDER, exist_ok=True)

# Eliminar cualquier referencia a DB_PATH de SQLite, ya no se usará
# DB_PATH = os.path.join(TEMP_PROCESSING_FOLDER, "logs/interactions.db")
# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "leo-trainer-videos")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-west-2")

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION_NAME
)

transcribe_client = boto3.client(
    'transcribe',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION_NAME
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6378/0")

celery_app = Celery(
    'leo_trainer_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Mexico_City',
    enable_utc=False,
)

# --- Configuración de conexión a PostgreSQL (similar a app.py) ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set!")

def get_db_connection():
    parsed_url = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        database=parsed_url.path[1:],
        user=parsed_url.username,
        password=parsed_url.password,
        host=parsed_url.hostname,
        port=parsed_url.port,
        sslmode='require'
    )
    return conn

# --- init_db() para PostgreSQL (¡CORREGIDO: Usando get_db_connection()!) ---
def init_db():
    conn = None
    try:
        conn = get_db_connection() # ¡CORREGIDO!
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT,
                scenario TEXT,
                message TEXT,
                response TEXT,
                audio_path TEXT,
                timestamp TEXT,
                evaluation TEXT,
                evaluation_rh TEXT,
                duration_seconds INTEGER DEFAULT 0,
                tip TEXT,
                visual_feedback TEXT
            );
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE,
                start_date TEXT,
                end_date TEXT,
                active INTEGER DEFAULT 1,
                token TEXT UNIQUE
            );
        ''')
        conn.commit()
        print("\U0001F4C3 Database initialized or already exists in worker (PostgreSQL).")
    except Exception as e:
        print(f"\U0001F525 Error initializing PostgreSQL database in worker: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- patch_db_schema() para PostgreSQL (¡CORREGIDO: Usando get_db_connection()!) ---
def patch_db_schema():
    conn = None
    try:
        conn = get_db_connection() # ¡CORREGIDO!
        c = conn.cursor()

        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='interactions' AND column_name='evaluation_rh'")
        if not c.fetchone():
            c.execute("ALTER TABLE interactions ADD COLUMN evaluation_rh TEXT;")
            print("Added 'evaluation_rh' to interactions table in worker.")

        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='interactions' AND column_name='tip'")
        if not c.fetchone():
            c.execute("ALTER TABLE interactions ADD COLUMN tip TEXT;")
            print("Added 'tip' to interactions table in worker.")

        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='interactions' AND column_name='visual_feedback'")
        if not c.fetchone():
            c.execute("ALTER TABLE interactions ADD COLUMN visual_feedback TEXT;")
            print("Added 'visual_feedback' to interactions table in worker.")
            
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='token'")
        if not c.fetchone():
            c.execute("ALTER TABLE users ADD COLUMN token TEXT UNIQUE;")
            print("Added 'token' to users table in worker.")

        conn.commit()
        print("\U0001F527 Database schema patched in worker (PostgreSQL).")
    except Exception as e:
        print(f"\U0001F525 Error patching PostgreSQL database schema in worker: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# Ejecutar la inicialización y parcheo de la DB al inicio del worker
init_db()
patch_db_schema()

def upload_file_to_s3(file_path, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket, object_name) 
        print(f"[S3 UPLOAD] Archivo {file_path} subido a s3://{bucket}/{object_name}")
        return f"https://{bucket}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{object_name}"
    except ClientError as e:
        print(f"[S3 ERROR] Falló la subida a S3: {e}")
        return None

def download_file_from_s3(bucket, object_name, file_path):
    try:
        s3_client.download_file(bucket, object_name, file_path)
        print(f"[S3 DOWNLOAD] Archivo s3://{bucket}/{object_name} descargado a {file_path}")
        return True
    except ClientError as e:
        print(f"[S3 ERROR] Falló la descarga de S3: {e}")
        return False

def convert_webm_to_mp4(input_path, output_path):
    try:
        command = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-strict", "experimental", "-y",
            output_path
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f"[FFMPEG] Conversion successful: {input_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG ERROR] Conversion failed for {input_path}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("[FFMPEG ERROR] ffmpeg not found. Please ensure it's installed and in your PATH.")
        return False
    except Exception as e:
        print(f"[FFMPEG ERROR] Unexpected error during conversion: {e}")
        return False

# La función analyze_video_posture está bien, ya devuelve 3 valores.
# No necesita cambios aquí si ya funciona como esperaba.
def analyze_video_posture(video_path):
    # Ya no importamos mediapipe globalmente. Si se usa, debe importarse aquí.
    # Si mp.solutions.face_detection no se usa o falla, cv2.CascadeClassifier se usará como fallback.
    # Removí `mp_face = mp.solutions.face_detection` del cuerpo principal si no se usa.

    summary = {"frames_total": 0, "face_detected_frames": 0, "error": None}
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            summary["error"] = "Could not open video file."
            print(f"[ERROR] analyze_video_posture: {summary['error']} {video_path}")
            return "⚠️ No se pudo abrir el archivo de video para análisis visual.", "Error en video", "N/A" # Devuelve 3 valores

        # Usar cv2.CascadeClassifier para detección facial, es más robusto sin dependencias de hardware
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frames_to_analyze = min(200, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) 

        if frames_to_analyze == 0:
            cap.release()
            return "⚠️ No se encontraron frames para analizar en el video.", "Sin frames", "0.0%" # Devuelve 3 valores

        for _ in range(frames_to_analyze):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                summary["face_detected_frames"] += 1
        cap.release()
        
        if summary["frames_total"] > 0:
            ratio = summary["face_detected_frames"] / summary["frames_total"]
        else:
            ratio = 0

        if ratio >= 0.7:
            return "✅ Te mostraste visible y profesional frente a cámara.", "Correcta", f"{ratio*100:.1f}%"
        elif ratio > 0:
            return "⚠️ Asegúrate de mantenerte visible durante toda la sesión.", "Mejorar visibilidad", f"{ratio*100:.1f}%"
        else:
            return "❌ No se detectó rostro en el video.", "No detectado", "0.0%"
    except Exception as e:
        return f"⚠️ Error en análisis visual: {str(e)}", "Error", "N/A" # Devuelve 3 valores


def compress_video_for_ai(input_path, output_path):
    try:
        command = [
            "ffmpeg", "-i", input_path,
            "-vf", "scale=160:120,format=gray",
            "-c:v", "libx264", "-crf", "32", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "32k", "-ac", "1",
            "-y",   # overwrite
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f"[COMPRESS] Video reducido: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] compress_video_for_ai: {e.stderr.decode()}")
        return False

@celery_app.task
def process_session_video(data):
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    duration = int(data.get("duration", 0))
    video_object_key = data.get("video_object_key")

    timestamp = datetime.now().isoformat()
    transcribed_text = ""
    public_summary = "Evaluación no disponible."
    internal_summary = "Evaluación no disponible."
    tip_text = "Consejo no disponible."
    posture_feedback = "Análisis visual no realizado." # Inicializamos para el caso de no video
    final_video_s3_url = None

    if not video_object_key:
        print("[ERROR] process_session_video: No se recibió video_object_key.")
        # Aseguramos que posture_feedback tenga un valor si no hay video
        posture_feedback = "Análisis visual no realizado (video no proporcionado)." 
        return {"status": "error", "error": "No se recibió el nombre del objeto de video de S3."}

    local_webm_path = os.path.join(TEMP_PROCESSING_FOLDER, video_object_key)
    
    if not os.path.exists(local_webm_path):
        print(f"[WARNING] Local WEBM file not found at {local_webm_path}. Attempting to download from S3 (assuming it was uploaded there).")
        if not download_file_from_s3(AWS_S3_BUCKET_NAME, video_object_key, local_webm_path):
            print(f"[ERROR] process_session_video: No se pudo encontrar/descargar el video WEBM: {video_object_key}")
            local_webm_path = None
            final_video_s3_url = "Video_Not_Available_Error"
            posture_feedback = "Análisis visual no realizado (video no disponible)." # Actualizamos si la descarga falla
    
    local_mp4_path = None
    temp_audio_path = None
    s3_audio_key = None

    if local_webm_path: # Solo proceder si el webm está disponible o se descargó
        mp4_object_key = video_object_key.replace('.webm', '.mp4')
        local_mp4_path = os.path.join(TEMP_PROCESSING_FOLDER, mp4_object_key)
        
        audio_filename_base = os.path.splitext(video_object_key)[0]
        s3_audio_key = f"audio/{audio_filename_base}.wav"
        temp_audio_path = os.path.join(TEMP_PROCESSING_FOLDER, f"{audio_filename_base}.wav")


        print(f"[INFO] Processing local video: {local_webm_path}")
        if convert_webm_to_mp4(local_webm_path, local_mp4_path):
            video_to_process_path = local_mp4_path
            print(f"[INFO] Converted to local MP4: {local_mp4_path}")

            compressed_path = local_mp4_path.replace(".mp4", "_compressed.mp4")
            if compress_video_for_ai(local_mp4_path, compressed_path):
                video_to_process_path = compressed_path
                print(f"[INFO] Compressed video ready: {compressed_path}")
            else:
                print(f"[WARNING] Compression failed. Using uncompressed MP4: {local_mp4_path}")

        else:
            video_to_process_path = local_webm_path
            print(f"[WARNING] Failed to convert to MP4, attempting to process original WEBM: {local_webm_path}")
            posture_feedback = "Análisis visual no realizado (falló la conversión de video)." # Actualizamos si la conversión falla

        try:
            print("[INFO] Extracting audio from video using ffmpeg directly...")
            command = [
                "ffmpeg", "-i", video_to_process_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                temp_audio_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[INFO] Audio extracted to {temp_audio_path}.")

            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                audio_s3_url = upload_file_to_s3(temp_audio_path, AWS_S3_BUCKET_NAME, s3_audio_key)
                if not audio_s3_url:
                    raise Exception("Failed to upload audio to S3 for Transcribe.")
                print(f"[S3 UPLOAD] Audio subido a S3: {audio_s3_url}")

                job_name = f"leo-trainer-transcription-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
                transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': audio_s3_url},
                    MediaFormat='wav',
                    LanguageCode='es-US'
                )
                print(f"[AWS TRANSCRIBE] Transcripción iniciada: {job_name}")

                max_attempts = 60
                for attempt in range(max_attempts):
                    status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                    job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                    if job_status in ['COMPLETED', 'FAILED']:
                        break
                    print(f"[AWS TRANSCRIBE] Estado de la transcripción: {job_status}. Esperando...")
                    time.sleep(10)

                if job_status == 'COMPLETED':
                    transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    transcript_response = requests.get(transcript_uri)
                    transcript_json = transcript_response.json()
                    transcribed_text = transcript_json['results']['transcripts'][0]['transcript']
                    print(f"[AWS TRANSCRIBE] Transcripción completa: '{transcribed_text}'")
                else:
                    print(f"[AWS TRANSCRIBE ERROR] Transcripción fallida o no completada: {job_status}")
                    transcribed_text = "Error en transcripción (servicio AWS Transcribe)."

            else:
                print("⚠️ No se pudo extraer audio o el archivo de audio está vacío.")
                transcribed_text = "Error en transcripción (audio vacío o no extraído)."
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error durante la extracción de audio con ffmpeg: {e.stderr.decode()}")
            transcribed_text = "Error en transcripción (extracción de audio fallida)."
        except Exception as e:
            print(f"[ERROR] Error durante la transcripción con AWS Transcribe: {e}")
            transcribed_text = "Error en transcripción (servicio AWS Transcribe)."

        # Llamada a analyze_video_posture (si hay un video para procesar)
        if video_to_process_path and os.path.exists(video_to_process_path):
            try:
                print(f"[VIDEO ANALYSIS] Starting posture analysis for: {video_to_process_path}")
                # analyze_video_posture devuelve 3 valores (feedback público, eval interna, porcentaje)
                visual_feedback_public, visual_eval_internal, visual_pct = analyze_video_posture(video_to_process_path)
                posture_feedback = visual_feedback_public # Usar el feedback público
                print(f"[POSTURA] {posture_feedback}")
            except Exception as e:
                posture_feedback = f"⚠️ Error inesperado en análisis visual: {str(e)}"
                print(f"[ERROR] Unexpected error in visual analysis: {e}")
        else:
            # Esto ya debería estar cubierto por los checks de local_webm_path al principio
            # pero lo mantenemos para claridad
            posture_feedback = "Análisis visual no realizado (video no disponible para análisis)."

        # Subir MP4 final a S3
        if local_mp4_path and os.path.exists(local_mp4_path):
            final_video_s3_url = upload_file_to_s3(local_mp4_path, AWS_S3_BUCKET_NAME, mp4_object_key)
            if not final_video_s3_url:
                print(f"[ERROR] Falló la subida del MP4 final a S3: {local_mp4_path}")
        else:
            final_video_s3_url = "Video_Processing_Failed"
            print(f"[WARNING] No se generó archivo MP4 local para subir a S3: {local_mp4_path}")

    else: # Si local_webm_path era None desde el principio (video_object_key faltante o descarga fallida)
        print("[WARNING] Skipping all video processing (conversion, transcription, visual analysis, S3 upload) due to missing WEBM file.")
        transcribed_text = "Transcripción no disponible (video faltante)."
        posture_feedback = "Análisis visual no realizado (video faltante)."
        final_video_s3_url = "Video_Missing_Error"

    if s3_audio_key:
        try:
            s3_client.delete_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_audio_key)
            print(f"[CLEANUP] Archivo de audio temporal de S3 eliminado: {s3_audio_key}")
        except ClientError as e:
            print(f"[CLEANUP ERROR] Falló la eliminación del archivo de audio de S3 {s3_audio_key}: {e}")

    final_user_text = transcribed_text
    leo_dialogue = "" # Ya no se usa la conversación de D-ID

    full_conversation_text = f"Participante: {final_user_text}\nLeo: (No se capturó el diálogo del agente D-ID)"

    print(f"DEBUG: final_user_text antes de evaluate_interaction: '{final_user_text}'")

    if not final_user_text.strip():
        print("[ERROR] No hay texto de usuario para evaluar (transcripción de AWS Transcribe vacía).")
        public_summary = "Error: No hay contenido de usuario para evaluar."
        internal_summary_db = json.dumps({"error": "No hay contenido de usuario para evaluar (transcripción vacía)."}) # Aseguramos formato JSON
        tip_text = "Asegúrate de que tu micrófono funcione y hables claramente durante la sesión."
    else:
        try:
            summaries = evaluate_interaction(final_user_text, leo_dialogue, video_to_process_path if video_to_process_path and os.path.exists(video_to_process_path) else None)
            
            print(f"DEBUG: Tipo de summaries después de evaluate_interaction: {type(summaries)}")
            print(f"DEBUG: Contenido de summaries después de evaluate_interaction: {summaries}")

            public_summary = summaries.get("public", public_summary)
            internal_summary_dict = summaries.get("internal", {})
            if "error" in internal_summary_dict:
                public_summary = "⚠️ Evaluación automática no disponible (error interno)."
                internal_summary_db = json.dumps({"error": internal_summary_dict.get("error", "Error desconocido durante la evaluación AI.")})
            else:
                internal_summary_db = json.dumps(internal_summary_dict, ensure_ascii=False, indent=2)
            
            print(f"[INFO] AI Evaluation successful. Public: {public_summary[:50]}...")
        except Exception as e:
            public_summary = "⚠️ Evaluación automática no disponible (falló la llamada a la IA)."
            internal_summary_db = json.dumps({"error": f"Error al llamar a evaluate_interaction: {str(e)}"})
            print(f"[ERROR] Error calling evaluate_interaction: {e}")

        try:
            from openai import OpenAI
            openai_client_for_tip = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            tip_completion = openai_client_for_tip.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Eres un coach médico empático y útil. Ofrece 2-3 consejos prácticos, claros y concretos sobre cómo mejorar las interacciones de un representante médico con doctores. Enfócate en el participante."},
                    {"role": "user", "content": f"Basado en la transcripción del participante:\n\nParticipante: {final_user_text}\n\n¿Qué podría hacer mejor el participante la próxima vez en una interacción con un doctor? Ofrece consejos accionables y positivos, asumiendo que el doctor (Leo) es un avatar interactivo cuyo diálogo no se evalúa directamente."}
                ],
                temperature=0.7,
            )
            tip_text = tip_completion.choices[0].message.content.strip()
            print(f"[INFO] AI Tip generated: {tip_text[:50]}...")
        except Exception as e:
            tip_text = f"⚠️ No se pudo generar un consejo automático: {str(e)}"
            print(f"[ERROR] Error generating personalized tip: {e}")

    conn = None # Inicializar conn para el bloque finally
    try:
        conn = get_db_connection() # ¡CORREGIDO: Usando get_db_connection() para PostgreSQL!
        c = conn.cursor()
        
        # Usamos INSERT con %s para PostgreSQL
        c.execute("""INSERT INTO interactions (
                name, email, scenario, message, response, timestamp,
                evaluation, evaluation_rh, duration_seconds, tip,
                audio_path, visual_feedback
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
                name, email, scenario, full_conversation_text, leo_dialogue, timestamp,
                public_summary, internal_summary_db, duration, tip_text,
                final_video_s3_url, posture_feedback
            ))
        conn.commit()
        print(f"[DB] Inserted new interaction for {email}, scenario {scenario} in PostgreSQL.")

    except Exception as e:
        print(f"[ERROR] Database save failed in Celery task (PostgreSQL): {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    temp_files_to_clean = [local_webm_path, local_mp4_path, temp_audio_path]
    if 'compressed_path' in locals() and compressed_path and os.path.exists(compressed_path):
        temp_files_to_clean.append(compressed_path)

    for temp_file in temp_files_to_clean:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"[CLEANUP] Archivo temporal eliminado: {temp_file}")
            except Exception as e:
                print(f"[CLEANUP ERROR] Falló la eliminación del archivo temporal {temp_file}: {e}")

    return {
        "status": "ok",
        "evaluation": public_summary,
        "tip": tip_text,
        "visual_feedback": posture_feedback,
        "final_video_url": final_video_s3_url,
        "full_conversation_text": full_conversation_text,
        "leo_dialogue": leo_dialogue,
        "timestamp": timestamp,
        "internal_summary": internal_summary_db,
        "duration": duration,
        "name": name,
        "email": email
    }