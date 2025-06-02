# celery_worker.py
import os
import sqlite3
from datetime import datetime
from celery import Celery
from dotenv import load_dotenv

# Imports for video processing and AI evaluation, now in the worker
from evaluator import evaluate_interaction
from moviepy.editor import VideoFileClip
import cv2
import mediapipe as mp
import subprocess
import json

# Import boto3 for S3 operations, as the worker will now manage S3 uploads/downloads
import boto3
from botocore.exceptions import ClientError

load_dotenv()

# --- Configuration for Database and AWS S3 (duplicated from app.py for worker self-sufficiency) ---
# IMPORTANT: Ensure these match your app.py and environment variables
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

# Load Whisper model once when the worker starts
try:
    import whisper
    whisper_model = whisper.load_model("base")
    print("\U0001F3A7 Whisper model loaded successfully in Celery worker.")
except Exception as e:
    print(f"\U0001F525 Error loading Whisper model in Celery worker: {e}")
    whisper_model = None


# Replace with your Render Redis internal URL
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

# ----------------------
# Helper Functions (moved from app.py, now part of worker's scope)
# ----------------------
# Función para subir archivo a S3
def upload_file_to_s3(file_path, bucket, object_name=None):
    """Sube un archivo a un bucket de S3"""
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket, object_name)
        print(f"[S3 UPLOAD] Archivo {file_path} subido a s3://{bucket}/{object_name}")
        return f"https://{bucket}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{object_name}"
    except ClientError as e:
        print(f"[S3 ERROR] Falló la subida a S3: {e}")
        return None

# Función para descargar archivo de S3
def download_file_from_s3(bucket, object_name, file_path):
    """Descarga un archivo de un bucket de S3"""
    try:
        s3_client.download_file(bucket, object_name, file_path)
        print(f"[S3 DOWNLOAD] Archivo s3://{bucket}/{object_name} descargado a {file_path}")
        return True
    except ClientError as e:
        print(f"[S3 ERROR] Falló la descarga de S3: {e}")
        return False

def convert_webm_to_mp4(input_path, output_path):
    """Converts a .webm video to .mp4 using ffmpeg."""
    try:
        command = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-strict", "experimental", "-y", # -y to overwrite output files
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
    """Analyzes video for face detection as a proxy for posture/presence."""
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
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                summary["face_detected_frames"] += 1
        cap.release()
        
    except Exception as e:
        summary["error"] = str(e)
        print(f"[ERROR] Error during video posture analysis: {e}")
    return summary


# ----------------------
# Celery Task
# ----------------------
@celery_app.task
def process_session_video(data):
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    # conversation is passed as empty array from frontend now, as D-ID events are not captured
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
    
    local_mp4_path = None # Initialize outside if block
    temp_audio_path = None # Initialize outside if block

    if local_webm_path:
        mp4_object_key = video_object_key.replace('.webm', '.mp4')
        local_mp4_path = os.path.join(TEMP_PROCESSING_FOLDER, mp4_object_key)
        temp_audio_path = os.path.join(TEMP_PROCESSING_FOLDER, f"audio_from_{video_object_key.split('.')[0]}.wav")

        print(f"[INFO] Processing local video: {local_webm_path}")
        if convert_webm_to_mp4(local_webm_path, local_mp4_path):
            video_to_process_path = local_mp4_path
            print(f"[INFO] Converted to local MP4: {local_mp4_path}")
        else:
            video_to_process_path = local_webm_path
            print(f"[WARNING] Failed to convert to MP4, attempting to process original WEBM: {local_webm_path}")

        video_clip = None
        try:
            if whisper_model:
                try:
                    video_clip = VideoFileClip(video_to_process_path)
                    if video_clip.audio:
                        print("[INFO] Extracting audio from video...")
                        video_clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                        transcription_result = whisper_model.transcribe(temp_audio_path)
                        transcribed_text = transcription_result.get("text", "").strip()
                        print(f"[INFO] Transcripción de audio exitosa: {transcribed_text[:100]}...")
                    else:
                        print("⚠️ El video no tiene pista de audio para transcribir.")
                except Exception as e:
                    print(f"[ERROR] Error durante la extracción de audio o transcripción: {e}")
                    transcribed_text = "Error en transcripción."
                finally:
                    if video_clip:
                        video_clip.close()
            else:
                print("[WARNING] Whisper model not loaded, skipping audio transcription.")

            # --- VISUAL ANALYSIS ---
            if local_mp4_path and os.path.exists(local_mp4_path):
                try:
                    print(f"[VIDEO ANALYSIS] Starting posture analysis for: {local_mp4_path}")
                    video_stats = analyze_video_posture(local_mp4_path)
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
                print(f"[WARNING] Skipping visual analysis: Local MP4 file not found at {local_mp4_path}")
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


    # --- AI Evaluation (GPT) y Personalized Tip (GPT) ---
    # Now, final_user_text relies SOLELY on Whisper transcription.
    # leo_dialogue will be an empty string or placeholder as D-ID agent speech is not captured.
    final_user_text = transcribed_text # User speech comes from Whisper transcription of their video
    leo_dialogue = "" # Explicitly empty as D-ID agent speech is not needed for AI context
    
    # full_conversation_text can still be used for logging the raw conversation,
    # but we'll use a placeholder for Leo's side if it's not explicitly captured.
    # Since 'conversation' is empty, full_conversation_text will also be empty or contain only user speech
    # if it was ever manually added outside D-ID events. For now, it will be empty.
    full_conversation_text = f"Participante: {final_user_text}\nLeo: (No se capturó el diálogo del agente D-ID)"


    if not final_user_text.strip(): # Check only final_user_text as leo_dialogue is now intentionally empty
        print("[ERROR] No hay texto de usuario para evaluar (transcripción de Whisper vacía).")
        public_summary = "Error: No hay contenido de usuario para evaluar."
        internal_summary = json.dumps({"error": "No hay contenido de usuario para evaluar."})
        tip_text = "Asegúrate de que tu micrófono funcione y hables claramente durante la sesión."
    else:
        try:
            # Pass final_user_text and an empty string for leo_dialogue to evaluator
            summaries = evaluate_interaction(final_user_text, leo_dialogue, local_mp4_path if local_mp4_path and os.path.exists(local_mp4_path) else None)
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
            # Generate tip based ONLY on user's performance and a generic "Leo's role"
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

    # --- SAVE TO DATABASE ---
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
                name, email, scenario, full_conversation_text, leo_dialogue, timestamp, # leo_dialogue is empty now
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

    # --- Clean up temporary local files ---
    temp_files_to_clean = [local_webm_path, local_mp4_path, temp_audio_path] if local_webm_path else []
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