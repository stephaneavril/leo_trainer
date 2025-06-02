import os
import sqlite3
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import openai
import whisper
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import evaluate_interaction # Assuming this is your evaluation logic
from moviepy.editor import VideoFileClip
import cv2
import mediapipe as mp
import subprocess # For ffmpeg
import secrets # For token generation
import json
from celery import Celery
from celery_worker import celery_app

# Importar boto3 para S3
import boto3
from botocore.exceptions import ClientError

print("\U0001F680 Iniciando Leo Virtual Trainer...")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai # This correctly sets the OpenAI client

# --- Configuración de AWS S3 ---
# Obtener las credenciales de AWS de las variables de entorno
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "leo-trainer-videos") # Usa el nombre de tu bucket de S3
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-west-2") # Usa la región de tu bucket de S3

# Inicializar cliente S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION_NAME
)

# Load Whisper model once when the app starts
try:
    whisper_model = whisper.load_model("base")
    print("\U0001F3A7 Whisper model loaded successfully.")
except Exception as e:
    print(f"\U0001F525 Error loading Whisper model: {e}")
    whisper_model = None # Handle case where model fails to load

# --- Configuration (Carpeta temporal local para procesamiento) ---
# Esta carpeta será usada solo para archivos temporales mientras se procesan.
# No será el almacenamiento permanente.
TEMP_PROCESSING_FOLDER = "/tmp/leo_trainer_processing/"
os.makedirs(TEMP_PROCESSING_FOLDER, exist_ok=True)

# Obtener la ruta del directorio actual donde se ejecuta app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- INICIALIZAR FLASK CON RUTAS DE PLANTILLAS Y ESTÁTICOS ---
# Flask busca 'templates' y 'static' en la misma carpeta que el archivo app.py
# Al especificar template_folder y static_folder, nos aseguramos de que Flask los encuentre.
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
CORS(app) # Enable CORS for frontend communication

@app.route("/")
def index():
    return render_template("index.html")

# app.config['UPLOAD_FOLDER'] ya no es el destino final, es solo temporal si se usa
app.config['UPLOAD_FOLDER'] = TEMP_PROCESSING_FOLDER

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-fallback") # Use a strong, unique key in .env
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") # Change this for production!

# DB_PATH ahora debe estar en un disco persistente o en PostgreSQL.
# Por ahora, lo mantenemos como SQLite local, pero recuerda que esto es un punto débil.
DB_PATH = os.path.join(TEMP_PROCESSING_FOLDER, "logs/interactions.db") # Mover a /tmp para evitar problemas de escritura si no hay disco
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ----------------------
# DB Init & Schema Patching
# ----------------------
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
        audio_path TEXT,        -- URL del video en S3
        timestamp TEXT,
        evaluation TEXT,        -- Public summary from GPT
        evaluation_rh TEXT,     -- Internal/RH summary from GPT
        duration_seconds INTEGER DEFAULT 0,
        tip TEXT,               -- Personalized tip from GPT
        visual_feedback TEXT    -- Feedback from video analysis
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
    print("\U0001F4C3 Database initialized or already exists.")

def patch_db_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check and add 'evaluation_rh' if not exists
    c.execute("PRAGMA table_info(interactions)")
    columns = [col[1] for col in c.fetchall()]
    if 'evaluation_rh' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN evaluation_rh TEXT")
        print("Added 'evaluation_rh' to interactions table.")

    # Check and add 'tip' if not exists
    if 'tip' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN tip TEXT")
        print("Added 'tip' to interactions table.")

    # Check and add 'visual_feedback' if not exists
    if 'visual_feedback' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN visual_feedback TEXT")
        print("Added 'visual_feedback' to interactions table.")

    # Check and add 'token' to users table if not exists
    c.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in c.fetchall()]
    if 'token' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN token TEXT UNIQUE")
        print("Added 'token' to users table.")
        # Optionally, generate tokens for existing users if needed:
        # c.execute("UPDATE users SET token = substr(hex(randomblob(8)), 1, 16) WHERE token IS NULL")

    conn.commit()
    conn.close()
    print("\U0001F527 Database schema patched.")

# Initialize and patch DB on app startup
init_db()
patch_db_schema()

# ----------------------
# Helper Functions
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
        # Using DEVNULL to suppress ffmpeg output unless there's an error
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

        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                summary["frames_total"] += 1
                # Convert the BGR image to RGB.
                results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    summary["face_detected_frames"] += 1
        cap.release()
    except Exception as e:
        summary["error"] = str(e)
        print(f"[ERROR] Error during video posture analysis: {e}")
    return summary

# ----------------------
# Celery Task (ahora maneja S3)
# ----------------------
@celery_app.task
def process_session_video(data):
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    duration = int(data.get("duration", 0))
    # video_object_key es el nombre del archivo en S3 (ej: "usuario_timestamp.webm")
    video_object_key = data.get("video_object_key")

    timestamp = datetime.now().isoformat()
    transcribed_text = ""
    public_summary = "Evaluación no disponible."
    internal_summary = "Evaluación no disponible."
    tip_text = "Consejo no disponible."
    posture_feedback = "Análisis visual no realizado."
    final_video_s3_url = None # Para almacenar la URL final del MP4 en S3

    if not video_object_key:
        print("[ERROR] process_session_video: No se recibió video_object_key.")
        return {"status": "error", "error": "No se recibió el nombre del objeto de video de S3."}

    # --- Descargar WEBM de S3 a una carpeta temporal local para procesamiento ---
    local_webm_path = os.path.join(TEMP_PROCESSING_FOLDER, video_object_key)
    if not download_file_from_s3(AWS_S3_BUCKET_NAME, video_object_key, local_webm_path):
        print(f"[ERROR] process_session_video: No se pudo descargar el video WEBM de S3: {video_object_key}")
        return {"status": "error", "error": f"No se pudo descargar el video WEBM de S3: {video_object_key}"}

    # --- Conversión y Extracción de Audio (ahora desde el archivo local) ---
    mp4_object_key = video_object_key.replace('.webm', '.mp4')
    local_mp4_path = os.path.join(TEMP_PROCESSING_FOLDER, mp4_object_key)
    temp_audio_path = os.path.join(TEMP_PROCESSING_FOLDER, f"audio_from_{video_object_key.split('.')[0]}.wav")

    print(f"[INFO] Processing local video: {local_webm_path}")
    if convert_webm_to_mp4(local_webm_path, local_mp4_path):
        video_to_process_path = local_mp4_path
        print(f"[INFO] Converted to local MP4: {local_mp4_path}")
    else:
        video_to_process_path = local_webm_path # Try processing original if conversion fails
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

        # --- AI Evaluation (GPT) y Personalized Tip (GPT) ---
        # (La lógica de evaluación y tip se mantiene igual, ya que usa el texto)
        user_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'user'])
        final_user_text = transcribed_text if transcribed_text.strip() else user_dialogue
        leo_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'agent'])
        if not leo_dialogue:
            leo_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'leo'])
        full_conversation_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in conversation])

        if not final_user_text.strip() and not leo_dialogue.strip():
            print("[ERROR] No hay texto de usuario ni de Leo para evaluar.")
            public_summary = "Error: No hay contenido para evaluar."
            tip_text = "Asegúrate de hablar y de que el asistente Leo también responda."
        else:
            try:
                summaries = evaluate_interaction(final_user_text, leo_dialogue)
                public_summary = summaries.get("public", public_summary)
                internal_summary = summaries.get("internal", internal_summary)
            except Exception as e:
                public_summary = "⚠️ Evaluación automática no disponible."
                internal_summary = f"❌ Error en evaluación: {str(e)}"
                print(f"[ERROR] Error calling evaluate_interaction: {e}")

            try:
                tip_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Eres un coach médico empático y útil. Ofrece 2-3 consejos prácticos, claros y concretos sobre cómo mejorar las interacciones de un representante médico con doctores. Enfócate en el participante."},
                        {"role": "user", "content": f"Basado en esta conversación:\n\nParticipante: {final_user_text}\nLeo: {leo_dialogue}\n\n¿Qué podría hacer mejor el participante la próxima vez? Ofrece consejos accionables y positivos."}
                    ],
                    temperature=0.7,
                )
                tip_text = tip_completion.choices[0].message.content.strip()
            except Exception as e:
                tip_text = f"⚠️ No se pudo generar un consejo automático: {str(e)}"
                print(f"[ERROR] Error generating personalized tip: {e}")

        # --- VISUAL ANALYSIS (desde el archivo local) ---
        if os.path.exists(local_mp4_path): # Usar el MP4 para el análisis visual si existe
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
        if os.path.exists(local_mp4_path):
            final_video_s3_url = upload_file_to_s3(local_mp4_path, AWS_S3_BUCKET_NAME, mp4_object_key)
            if not final_video_s3_url:
                print(f"[ERROR] Falló la subida del MP4 final a S3: {local_mp4_path}")
                return {"status": "error", "error": "Falló la subida del MP4 final a S3."}
        else:
            print(f"[WARNING] No se generó archivo MP4 local para subir a S3: {local_mp4_path}")
            # Si no hay MP4, ¿qué URL guardamos? Podríamos usar la del WEBM original si no se borra.
            # Por ahora, si no hay MP4, no guardamos URL de video final.
            final_video_s3_url = None # No hay MP4 para subir

    except Exception as general_error:
        print(f"[CRITICAL ERROR] Unhandled error in process_session_video task: {general_error}")
        return {"status": "error", "error": f"Internal server error during processing: {str(general_error)}"}
    finally:
        # --- Limpiar archivos temporales locales ---
        for temp_file in [local_webm_path, local_mp4_path, temp_audio_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"[CLEANUP] Archivo temporal eliminado: {temp_file}")
                except Exception as e:
                    print(f"[CLEANUP ERROR] Falló la eliminación del archivo temporal {temp_file}: {e}")

    # --- Retornar resultados para que el Web Service los guarde en la DB ---
    # La tarea ahora devuelve todos los datos relevantes, incluyendo la URL final del video en S3
    return {
        "status": "ok",
        "evaluation": public_summary,
        "tip": tip_text,
        "visual_feedback": posture_feedback,
        "final_video_url": final_video_s3_url, # URL del video MP4 en S3
        "full_conversation_text": full_conversation_text,
        "leo_dialogue": leo_dialogue,
        "timestamp": timestamp,
        "internal_summary": internal_summary,
        "duration": duration,
        "name": name, # Incluir para que el Web Service pueda identificar el registro
        "email": email # Incluir para que el Web Service pueda identificar el registro
    }


# ----------------------
# Rutas de la Aplicación (ahora interactúan con S3 y Celery)
# ----------------------

@app.route("/upload_video", methods=["POST"])
def upload_video():
    name = request.form.get("name")
    email = request.form.get("email")
    file = request.files.get("video")

    if not file:
        return jsonify({"error": "No se envió ningún archivo de video."}), 400
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    # El nombre del objeto en S3 (clave del objeto)
    webm_object_key = secure_filename(f"{name}_{email}_{timestamp_str}.webm")

    # Guardar el archivo temporalmente en el sistema de archivos local del Web Service
    # antes de subirlo a S3. Esta es una carpeta temporal /tmp.
    local_temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], webm_object_key)
    try:
        file.save(local_temp_filepath)
        print(f"[UPLOAD LOCAL] Video WEBM guardado temporalmente en: {local_temp_filepath}")
    except Exception as e:
        print(f"[ERROR] Falló el guardado local temporal del video: {e}")
        return jsonify({"status": "error", "error": f"Error al guardar el video temporalmente: {str(e)}"}), 500

    # Subir el archivo WEBM a S3 inmediatamente
    s3_url = upload_file_to_s3(local_temp_filepath, AWS_S3_BUCKET_NAME, webm_object_key)

    # Limpiar el archivo temporal local después de subir a S3
    try:
        if os.path.exists(local_temp_filepath):
            os.remove(local_temp_filepath)
            print(f"[CLEANUP LOCAL] Archivo temporal local eliminado: {local_temp_filepath}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Falló la eliminación del archivo temporal local {local_temp_filepath}: {e}")

    if not s3_url:
        return jsonify({"status": "error", "error": "Falló la subida del video a S3."}), 500

    # Retornar la clave del objeto S3 (el nombre del archivo en S3)
    return jsonify({"status": "saved", "path": webm_object_key})


@app.route("/log_full_session", methods=["POST"])
def log_full_session_api():
    data = request.get_json()
    # video_object_key es el nombre del archivo en S3 que se subió en /upload_video
    video_object_key = data.get("video_filename") # Renombrado para claridad

    if not video_object_key:
        return jsonify({"status": "error", "error": "No se recibió el nombre del objeto de video de S3."}), 400

    # Enqueue the task (la tarea ahora recibe la clave del objeto S3)
    task_data = {
        "name": data.get("name"),
        "email": data.get("email"),
        "scenario": data.get("scenario"),
        "conversation": data.get("conversation", []),
        "duration": int(data.get("duration", 0)),
        "video_object_key": video_object_key # Pasar la clave del objeto S3
    }
    task = process_session_video.apply_async(args=[task_data])
    print(f"[INFO] Task enqueued with ID: {task.id}")

    return jsonify({
        "status": "processing",
        "task_id": task.id,
        "message": "Sesión registrada, evaluación en progreso."
    })


@app.route("/task_status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task = process_session_video.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {"status": "pending", "message": "Task is pending..."}
    elif task.state == 'STARTED':
        response = {"status": "processing", "message": "Task is processing..."}
    elif task.state == 'SUCCESS':
        # Si la tarea fue exitosa, el resultado contiene todos los datos de la evaluación
        result_data = task.result
        # Guardar los resultados en la base de datos (SQLite)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("""INSERT INTO interactions (
                name, email, scenario, message, response, timestamp,
                evaluation, evaluation_rh, duration_seconds, tip,
                audio_path, visual_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                result_data.get("name"),
                result_data.get("email"),
                result_data.get("scenario"),
                result_data.get("full_conversation_text"),
                result_data.get("leo_dialogue"),
                result_data.get("timestamp"),
                result_data.get("evaluation"),
                result_data.get("internal_summary"),
                result_data.get("duration"),
                result_data.get("tip"),
                result_data.get("final_video_url"), # ¡Ahora guardamos la URL de S3!
                result_data.get("visual_feedback")
            ))
            conn.commit()
            print(f"[DB] Interaction saved for {result_data.get('email')}, scenario {result_data.get('scenario')}.")
            db_save_status = "ok" # Changed db_status to db_save_status to avoid conflict
        except Exception as e:
            print(f"[ERROR] Database insert failed from task_status: {e}")
            db_save_status = "error" # Changed db_status to db_save_status to avoid conflict
        finally:
            conn.close()

        response = {
            "status": "completed",
            "result": {
                "evaluation": result_data.get("evaluation"),
                "tip": result_data.get("tip"),
                "visual_feedback": result_data.get("visual_feedback"),
                "final_video_url": result_data.get("final_video_url"),
                "db_save_status": db_save_status # Changed db_status to db_save_status to avoid conflict
            }
        }
    elif task.state == 'FAILURE':
        response = {"status": "failed", "message": str(task.info)}
    else:
        response = {"status": task.state, "message": "Unknown state"}
    return jsonify(response)


@app.route("/video/<path:filename>") # Cambiar a path para manejar URLs completas si es necesario
def serve_video(filename):
    # Asumimos que 'filename' es ahora la URL completa del video en S3
    # o el nombre del objeto en S3.
    # Si es solo el nombre del objeto, construimos la URL pública de S3.
    # Si ya es una URL completa, la usamos directamente.
    
    # Para simplificar, asumimos que audio_path en la DB es la URL completa de S3
    # Si audio_path solo guarda el nombre del objeto, necesitarías construir la URL aquí:
    # video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{filename}"
    
    # Render puede tener problemas sirviendo directamente desde S3 vía redirect.
    # La forma más sencilla es que el frontend use la URL de S3 directamente.
    # Por ahora, esta ruta puede redirigir o simplemente retornar la URL de S3.
    
    # Opción 1: Redirigir directamente a la URL de S3 (más eficiente)
    s3_video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{filename}"
    print(f"[SERVE VIDEO] Redirigiendo a S3: {s3_video_url}")
    return redirect(s3_video_url, code=302) # Redirección temporal


# ----------------------
# Admin Routes
# ----------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    print(f"DEBUG: Accediendo a /admin. Método HTTP: {request.method}") # <--- AÑADIR ESTA LÍNEA

    if not session.get("admin"):
        print("DEBUG: No hay sesión de administrador, redirigiendo a /login") # <--- AÑADIR ESTA LÍNEA
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # POST: user management
    if request.method == "POST":
        print("DEBUG: Recibida solicitud POST en /admin") # <--- AÑADIR ESTA LÍNEA
        action = request.form.get("action")
        print(f"DEBUG: Acción del formulario: {action}") # <--- AÑADIR ESTA LÍNEA
        if action == "add":
            print("DEBUG: Intentando añadir usuario") # <--- AÑADIR ESTA LÍNEA
            name = request.form["name"]
            email = request.form["email"]
            start = request.form["start_date"]
            end = request.form["end_date"]
            token = secrets.token_hex(8) # Generate a new token
            print(f"DEBUG: Datos de usuario: {name}, {email}, {start}, {end}, {token}") # <--- AÑADIR ESTA LÍNEA
            try: # <--- AÑADIR ESTE BLOQUE TRY/EXCEPT
                c.execute("""INSERT OR REPLACE INTO users (name, email, start_date, end_date, active, token)
                                   VALUES (?, ?, ?, ?, 1, ?)""", (name, email, start, end, token))
                conn.commit()
                print(f"[ADMIN] Added/Updated user: {email}")
            except Exception as e:
                print(f"ERROR: Falló al insertar usuario en DB: {e}") # <--- AÑADIR ESTA LÍNEA
                conn.rollback()
                return f"Error al guardar usuario: {str(e)}", 500 # <--- OPCIONAL: DEVOLVER EL ERROR PARA DEPURACIÓN
        elif action == "toggle":
            print("DEBUG: Intentando activar/desactivar usuario") # <--- AÑADIR ESTA LÍNEA
            user_id = int(request.form["user_id"])
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (user_id,))
            print(f"[ADMIN] Toggled user active status: {user_id}")
        elif action == "regen_token":
            print("DEBUG: Intentando regenerar token") # <--- AÑADIR ESTA LÍNEA
            user_id = int(request.form["user_id"])
            new_token = secrets.token_hex(8)
            c.execute("UPDATE users SET token = ? WHERE id = ?", (new_token, user_id))
            print(f"[ADMIN] Regenerated token for user: {user_id}")
        conn.commit()

    # Load interactions
    # audio_path ahora contendrá la URL completa de S3
    c.execute("""SELECT name, email, scenario, message, response, audio_path, timestamp, evaluation, evaluation_rh, tip, visual_feedback
                 FROM interactions
                 ORDER BY timestamp DESC""")
    raw_data = c.fetchall()

    # Process data to parse JSON from evaluation_rh
    processed_data = []
    for row in raw_data:
        try:
            # Parse the JSON string from evaluation_rh (row[8])
            parsed_rh_evaluation = json.loads(row[8])
        except (json.JSONDecodeError, TypeError):
            # Handle cases where it's not valid JSON (e.g., old entries or errors)
            parsed_rh_evaluation = {"error": "Invalid JSON or old format", "raw_content": row[8]}

        # Convert the tuple to a list to make it mutable, then append the parsed data
        processed_row = list(row)
        processed_row[8] = parsed_rh_evaluation # Replace the raw string with the parsed JSON object
        processed_data.append(processed_row)

    # Load users
    c.execute("SELECT id, name, email, start_date, end_date, active, token FROM users")
    users = c.fetchall()

    # Calculate minutes used per user
    c.execute("""
        SELECT u.name, u.email, COALESCE(SUM(i.duration_seconds), 0) as used_secs
        FROM users u
        LEFT JOIN interactions i ON u.email = i.email
        GROUP BY u.name, u.email
    """)
    usage_rows = c.fetchall()

    usage_summaries = []
    total_minutes_all_users = 0
    for name_u, email_u, secs in usage_rows:
        mins = secs // 60
        total_minutes_all_users += mins
        summary = "Buen desempeño general" if mins >= 15 else "Actividad moderada" if mins >= 5 else "Poca actividad, se sugiere seguimiento"
        usage_summaries.append({
            "name": name_u,
            "email": email_u,
            "minutes": mins,
            "summary": summary
        })

    contracted_minutes = 1050 # Fixed: total contracted minutes for the admin

    conn.close()

    return render_template(
        "admin.html",
        data=processed_data, # Pass the processed data
        users=users,
        usage_summaries=usage_summaries,
        total_minutes=total_minutes_all_users,
        contracted_minutes=contracted_minutes
    )

@app.route("/end_session", methods=["POST", "GET"]) # Allow GET for simple redirection in some cases
def end_session_redirect():
    # This route is largely redundant now as the frontend's endSession handles logging
    # and redirects directly to dashboard after processing.
    # It can be used for a clean redirection if needed.
    name = session.get("name")
    email = session.get("email")
    if name and email:
        # Redirect to dashboard, passing name/email via form data for POST, or just rely on session for GET
        # For a clean redirect, it's better to pass them back to the dashboard route
        # which will then re-validate and pull from DB/session.
        return redirect(url_for('dashboard', name=name, email=email), code=307 if request.method == "POST" else 302)
    return redirect(url_for('index'))

@app.route("/admin/delete_session/<int:session_id>", methods=["POST"])
def delete_session(session_id):
    if not session.get("admin"):
        return redirect("/login") # Ensure only admins can delete

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. Get the video URL from the database
        c.execute("SELECT audio_path FROM interactions WHERE id = ?", (session_id,))
        row = c.fetchone()

        if row and row[0]: # Check if a record and audio_path (S3 URL) exist
            s3_video_url = row[0]
            # Extraer el nombre del objeto de S3 de la URL
            s3_object_key = s3_video_url.split('/')[-1]

            # 2. Eliminar el objeto de S3
            try:
                s3_client.delete_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_object_key)
                print(f"[DELETE S3] Successfully deleted S3 object: s3://{AWS_S3_BUCKET_NAME}/{s3_object_key}")
            except ClientError as e:
                print(f"[DELETE S3 ERROR] Failed to delete S3 object {s3_object_key}: {e}")
                # Si el archivo no existe en S3, no es un error crítico para la DB.
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"[DELETE S3 WARNING] S3 object not found at: {s3_object_key} (record will still be deleted)")
                else:
                    return f"Error al eliminar el video de S3: {str(e)}", 500

            # 3. Eliminar el registro de la base de datos
            c.execute("DELETE FROM interactions WHERE id = ?", (session_id,))
            conn.commit()
            print(f"[DB DELETE] Successfully deleted record for session ID: {session_id}")
            return redirect("/admin") # Redirect back to admin panel
        else:
            print(f"[DELETE ERROR] Session or video path not found for ID: {session_id}")
            return "Sesión no encontrada o sin video asociado.", 404

    except Exception as e:
        conn.rollback() # Rollback changes if an error occurs
        print(f"[DELETE ERROR] Failed to delete session {session_id}: {e}")
        return f"Error al eliminar la sesión: {str(e)}", 500
    finally:
        conn.close()

@app.route("/healthz")
def health_check():
    return "OK", 200
