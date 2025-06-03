# celery_worker.py

import os
import sqlite3
from datetime import datetime, date
from celery import Celery
from dotenv import load_dotenv
import time # Necesario para esperar la transcripción

from evaluator import evaluate_interaction
import cv2
import mediapipe as mp
import subprocess
import json
import secrets

import boto3
from botocore.exceptions import ClientError

load_dotenv()

PERSISTENT_DISK_MOUNT_PATH = os.getenv("PERSISTENT_DISK_MOUNT_PATH", "/var/data")
TEMP_PROCESSING_FOLDER = os.path.join(PERSISTENT_DISK_MOUNT_PATH, "leo_trainer_processing")
os.makedirs(TEMP_PROCESSING_FOLDER, exist_ok=True)

DB_PATH = os.path.join(TEMP_PROCESSING_FOLDER, "logs/interactions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

# Cliente de AWS Transcribe
transcribe_client = boto3.client(
    'transcribe',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION_NAME
)

# Eliminar la carga del modelo Whisper
# try:
#     import whisper
#     whisper_model = whisper.load_model("tiny")
#     print("\U0001F3A7 Whisper model loaded successfully in Celery worker.")
# except Exception as e:
#     print(f"\U0001F525 Error loading Whisper model in Celery worker: {e}")
#     whisper_model = None

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

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        scenario TEXT,
        message TEXT,           -- Full conversation text
        response TEXT,          -- Leo's responses consolidated
        audio_path TEXT,        -- URL del video en S3 (will be updated by Celery task)
        timestamp TEXT,
        evaluation TEXT,        -- Public summary from GPT (will be updated by Celery task)
        evaluation_rh TEXT,     -- Internal/RH summary from GPT (will be updated by Celery task)
        duration_seconds INTEGER DEFAULT 0,
        tip TEXT,               -- Personalized tip from GPT (will be updated by Celery task)
        visual_feedback TEXT    -- Feedback from video analysis (will be updated by Celery task)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        start_date TEXT,
        end_date TEXT,
        active INTEGER DEFAULT 1,
        token TEXT UNIQUE -- Unique token for user access
    )''')
    conn.commit()
    conn.close()
    print("\U0001F4C3 Database initialized or already exists in worker.")

def patch_db_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("PRAGMA table_info(interactions)")
    columns = [col[1] for col in c.fetchall()]
    if 'evaluation_rh' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN evaluation_rh TEXT")
        print("Added 'evaluation_rh' to interactions table in worker.")

    if 'tip' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN tip TEXT")
        print("Added 'tip' to interactions table in worker.")

    if 'visual_feedback' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN visual_feedback TEXT")
        print("Added 'visual_feedback' to interactions table in worker.")

    c.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in c.fetchall()]
    if 'token' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN token TEXT UNIQUE")
        print("Added 'token' to users table in worker.")

    conn.commit()
    conn.close()
    print("\U0001F527 Database schema patched in worker.")

init_db()
patch_db_schema()

def upload_file_to_s3(file_path, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        # LÍNEA MODIFICADA: Eliminado ExtraArgs={'ACL': 'public-read'}
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

def analyze_video_posture(video_path):
    mp_face = mp.solutions.face_detection
    summary = {"frames_total": 0, "face_detected_frames": 0, "error": None}
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            summary["error"] = "Could not open video file."
            print(f"[ERROR] analyze_video_posture: {summary['error']} {video_path}")
            return summary

        frame_count_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_analyze = min(200, frame_count_cap)
        
        if frames_to_analyze == 0:
            summary["error"] = "No frames to analyze in video."
            cap.release()
            return summary

        for _ in range(frames_to_analyze):
            ret, frame = cap.read()
            if not ret:
                break
            summary["frames_total"] += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Usar cv2.CascadeClassifier para detección facial
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectManyScales(gray, 1.3, 5) # Corregir a detectMultiScale
            if len(faces) > 0:
                summary["face_detected_frames"] += 1
        cap.release()
        
    except Exception as e:
        summary["error"] = str(e)
        print(f"[ERROR] Error during video posture analysis: {e}")
    return summary

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
    posture_feedback = "Análisis visual no realizado."
    final_video_s3_url = None

    if not video_object_key:
        print("[ERROR] process_session_video: No se recibió video_object_key.")
        return {"status": "error", "error": "No se recibió el nombre del objeto de video de S3."}

    local_webm_path = os.path.join(TEMP_PROCESSING_FOLDER, video_object_key)
    
    if not os.path.exists(local_webm_path):
        print(f"[WARNING] Local WEBM file not found at {local_webm_path}. Attempting to download from S3 (assuming it was uploaded there).")
        if not download_file_from_s3(AWS_S3_BUCKET_NAME, video_object_key, local_webm_path):
            print(f"[ERROR] process_session_video: No se pudo encontrar/descargar el video WEBM: {video_object_key}")
            local_webm_path = None
            final_video_s3_url = "Video_Not_Available_Error"
    
    local_mp4_path = None
    temp_audio_path = None # Ruta del archivo de audio temporal local
    s3_audio_key = None # Clave del archivo de audio en S3

    if local_webm_path:
        mp4_object_key = video_object_key.replace('.webm', '.mp4')
        local_mp4_path = os.path.join(TEMP_PROCESSING_FOLDER, mp4_object_key)
        
        # Generar un nombre único para el archivo de audio en S3
        audio_filename_base = os.path.splitext(video_object_key)[0]
        s3_audio_key = f"audio/{audio_filename_base}.wav" # Guardar audios en una subcarpeta 'audio/' en S3
        temp_audio_path = os.path.join(TEMP_PROCESSING_FOLDER, f"{audio_filename_base}.wav")


        print(f"[INFO] Processing local video: {local_webm_path}")
        if convert_webm_to_mp4(local_webm_path, local_mp4_path):
            video_to_process_path = local_mp4_path # Default to uncompressed MP4
            print(f"[INFO] Converted to local MP4: {local_mp4_path}")

            # Comprimir MP4 para reducir su tamaño antes de continuar
            compressed_path = local_mp4_path.replace(".mp4", "_compressed.mp4")
            if compress_video_for_ai(local_mp4_path, compressed_path):
                video_to_process_path = compressed_path # Use compressed path if successful
                print(f"[INFO] Compressed video ready: {compressed_path}")
            else:
                # If compression fails, video_to_process_path remains local_mp4_path (uncompressed MP4)
                print(f"[WARNING] Compression failed. Using uncompressed MP4: {local_mp4_path}")

        else: # This 'else' correctly aligns with the 'if convert_webm_to_mp4' statement
            video_to_process_path = local_webm_path # Fallback to original WEBM
            print(f"[WARNING] Failed to convert to MP4, attempting to process original WEBM: {local_webm_path}")

        try:
            # --- Extracción de Audio con FFmpeg y Transcripción con AWS Transcribe ---
            # Eliminar la lógica de Whisper y reemplazarla por AWS Transcribe
            # if whisper_model: # Bloque de Whisper eliminado
            try:
                # Usar subprocess para extraer audio directamente con ffmpeg
                print("[INFO] Extracting audio from video using ffmpeg directly...")
                command = [
                    "ffmpeg", "-i", video_to_process_path,  # Video de entrada
                    "-vn",                                  # No incluir video
                    "-acodec", "pcm_s16le",                 # Codec de audio PCM (16-bit, little-endian)
                    "-ar", "16000",                         # Tasa de muestreo de audio (16 kHz, común para voz)
                    "-ac", "1",                             # Un canal de audio (mono)
                    "-y",                                   # Sobreescribir archivo de salida sin preguntar
                    temp_audio_path                         # Archivo de audio de salida
                ]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"[INFO] Audio extracted to {temp_audio_path}.")

                if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                    # Subir el archivo de audio a S3 para AWS Transcribe
                    audio_s3_url = upload_file_to_s3(temp_audio_path, AWS_S3_BUCKET_NAME, s3_audio_key)
                    if not audio_s3_url:
                        raise Exception("Failed to upload audio to S3 for Transcribe.")
                    print(f"[S3 UPLOAD] Audio subido a S3: {audio_s3_url}")

                    # Iniciar el trabajo de transcripción con AWS Transcribe
                    job_name = f"leo-trainer-transcription-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
                    transcribe_client.start_transcription_job(
                        TranscriptionJobName=job_name,
                        Media={'MediaFileUri': audio_s3_url},
                        MediaFormat='wav', # Asegúrate de que coincida con el formato de salida de ffmpeg
                        LanguageCode='es-MX' # O el idioma que necesites, ej. 'en-US'
                    )
                    print(f"[AWS TRANSCRIBE] Transcripción iniciada: {job_name}")

                    # Esperar a que el trabajo de transcripción finalice
                    max_attempts = 60 # Esperar hasta 10 minutos (60 * 10 segundos)
                    for attempt in range(max_attempts):
                        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                        if job_status in ['COMPLETED', 'FAILED']:
                            break
                        print(f"[AWS TRANSCRIBE] Estado de la transcripción: {job_status}. Esperando...")
                        time.sleep(10) # Esperar 10 segundos antes de la siguiente verificación

                    if job_status == 'COMPLETED':
                        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                        # Descargar el contenido del transcript
                        transcript_response = requests.get(transcript_uri)
                        transcript_json = transcript_response.json()
                        transcribed_text = transcript_json['results']['transcripts'][0]['transcript']
                        print(f"[AWS TRANSCRIBE] Transcripción completa: {transcribed_text[:100]}...")
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
            # Eliminar el bloque 'else: print("[WARNING] Whisper model not loaded...")'

            # --- VISUAL ANALYSIS ---
            if video_to_process_path and os.path.exists(video_to_process_path):
                try:
                    print(f"[VIDEO ANALYSIS] Starting posture analysis for: {video_to_process_path}")
                    video_stats = analyze_video_posture(video_to_process_path)
                    if video_stats["error"]:
                        posture_feedback = f"⚠️ Error en análisis visual: {video_stats['error']}"
                    elif video_stats['frames_total'] > 0:
                        visible_pct = (video_stats['face_detected_frames'] / video_stats['frames_total']) * 100
                        posture_feedback = f"Rostro visible en {visible_pct:.1f}% de los frames."
                    else:
                        posture_feedback = "No se pudieron analizar frames de video."
                    print(f"[POSTURA] {posture_feedback}")
                except Exception as e:
                    posture_feedback = f"⚠️ Error inesperado en análisis visual: {str(e)}"
                    print(f"[ERROR] Unexpected error in visual analysis: {e}")
            else:
                print(f"[WARNING] Skipping visual analysis: Processed video file not found at {video_to_process_path}")
                posture_feedback = "Video no disponible para análisis visual."

            # --- Subir MP4 final a S3 ---
            if local_mp4_path and os.path.exists(local_mp4_path):
                final_video_s3_url = upload_file_to_s3(local_mp4_path, AWS_S3_BUCKET_NAME, mp4_object_key)
                if not final_video_s3_url:
                    print(f"[ERROR] Falló la subida del MP4 final a S3: {local_mp4_path}")
            else:
                print(f"[WARNING] No se generó archivo MP4 local para subir a S3: {local_mp4_path}")
                final_video_s3_url = "Video_Processing_Failed"

        except Exception as video_processing_error:
            print(f"[CRITICAL ERROR] Error during video processing or S3 upload: {video_processing_error}")
            posture_feedback = f"Error general de procesamiento de video: {str(video_processing_error)}"
            final_video_s3_url = "Video_Processing_Error"
            
    else:
        print("[WARNING] Skipping all video processing (conversion, transcription, visual analysis, S3 upload) due to missing WEBM file.")
        transcribed_text = "Transcripción no disponible (video faltante)."
        posture_feedback = "Análisis visual no realizado (video faltante)."
        final_video_s3_url = "Video_Missing_Error"

    # Limpiar el archivo de audio de S3 después de la transcripción
    if s3_audio_key:
        try:
            s3_client.delete_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_audio_key)
            print(f"[CLEANUP] Archivo de audio temporal de S3 eliminado: {s3_audio_key}")
        except ClientError as e:
            print(f"[CLEANUP ERROR] Falló la eliminación del archivo de audio de S3 {s3_audio_key}: {e}")

    final_user_text = transcribed_text
    leo_dialogue = ""
    
    full_conversation_text = f"Participante: {final_user_text}\nLeo: (No se capturó el diálogo del agente D-ID)"

    if not final_user_text.strip():
        print("[ERROR] No hay texto de usuario para evaluar (transcripción de AWS Transcribe vacía).")
        public_summary = "Error: No hay contenido de usuario para evaluar."
        internal_summary = json.dumps({"error": "No hay contenido de usuario para evaluar."})
        tip_text = "Asegúrate de que tu micrófono funcione y hables claramente durante la sesión."
    else:
        try:
            summaries = evaluate_interaction(final_user_text, leo_dialogue, video_to_process_path if video_to_process_path and os.path.exists(video_to_process_path) else None)
            public_summary = summaries.get("public", public_summary)
            internal_summary = summaries.get("internal", internal_summary)
            if "error" in summaries.get("internal", {}):
                public_summary = "⚠️ Evaluación automática no disponible."
                internal_summary = json.dumps({"error": summaries.get("internal", {}).get("error", "Error desconocido en evaluación.")})
            print(f"[INFO] AI Evaluation successful. Public: {public_summary[:50]}...")
        except Exception as e:
            public_summary = "⚠️ Evaluación automática no disponible."
            internal_summary = json.dumps({"error": f"Error al llamar a evaluate_interaction: {str(e)}"})
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

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT id FROM interactions WHERE audio_path = ?", (video_object_key,))
        existing_row = c.fetchone()

        if existing_row:
            interaction_id = existing_row[0]
            c.execute("""UPDATE interactions SET
                name = ?, email = ?, scenario = ?, message = ?, response = ?, timestamp = ?,
                evaluation = ?, evaluation_rh = ?, duration_seconds = ?,
                tip = ?, visual_feedback = ?, audio_path = ?
                WHERE id = ?""", (
                name, email, scenario, full_conversation_text, leo_dialogue, timestamp,
                public_summary, internal_summary, duration, tip_text,
                posture_feedback, final_video_s3_url, interaction_id
            ))
            print(f"[DB] Updated interaction {interaction_id} for {email}, scenario {scenario}.")
        else:
            print(f"[WARNING] No preliminary DB entry found for {video_object_key}. Inserting new record.")
            c.execute("""INSERT INTO interactions (
                name, email, scenario, message, response, timestamp,
                evaluation, evaluation_rh, duration_seconds, tip,
                audio_path, visual_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                name, email, scenario, full_conversation_text, leo_dialogue, timestamp,
                public_summary, internal_summary, duration, tip_text,
                final_video_s3_url, posture_feedback
            ))
            print(f"[DB] Inserted new interaction for {email}, scenario {scenario}.")

        conn.commit()
    except Exception as e:
        print(f"[ERROR] Database save failed in Celery task: {e}")
        conn.rollback()
    finally:
        conn.close()

    temp_files_to_clean = [local_webm_path, local_mp4_path, temp_audio_path]
    # Add compressed_path to cleanup if it was created
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
        "internal_summary": internal_summary,
        "duration": duration,
        "name": name,
        "email": email
    }
