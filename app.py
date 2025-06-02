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
# Removed synchronous processing imports from app.py
# from evaluator import evaluate_interaction # This is now exclusively in celery_worker.py
# from moviepy.editor import VideoFileClip
# import cv2
# import mediapipe as mp
# import subprocess # For ffmpeg
import secrets
import json
from celery import Celery
from celery_worker import celery_app # Ensure this import is correct

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
    # Whisper model can be loaded in app.py if it's also used here,
    # but if it's only used by Celery task, move this to celery_worker.py
    # For now, keeping it here as it was in original to avoid breaking existing flow.
    # Best practice is to load it only where it's used (i.e., in celery_worker.py)
    whisper_model = whisper.load_model("base")
    print("\U0001F3A7 Whisper model loaded successfully.")
except Exception as e:
    print(f"\U0001F525 Error loading Whisper model: {e}")
    whisper_model = None # Handle case where model fails to load

# --- Configuration (Carpeta temporal local para procesamiento) ---
# Esta carpeta será usada solo para archivos temporales mientras se procesan.
# DEBE SER PERSISTENTE EN RENDER.
PERSISTENT_DISK_MOUNT_PATH = "/var/data" # Your Render persistent disk mount path
TEMP_PROCESSING_FOLDER = os.path.join(PERSISTENT_DISK_MOUNT_PATH, "leo_trainer_processing") # Use a subfolder on the persistent disk
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

@app.before_request
def log_request_info():
    print(f"DEBUG_HOOK: Request received: {request.method} {request.path}")
    if request.method == 'POST':
        print(f"DEBUG_HOOK: Form data: {request.form}")
        print(f"DEBUG_HOOK: Files: {request.files}")
        try:
            print(f"DEBUG_HOOK: JSON data: {request.json}")
        except Exception:
            print("DEBUG_HOOK: No JSON data or invalid JSON")

@app.route("/")
def index():
    return render_template("index.html")

# app.config['UPLOAD_FOLDER'] ya no es el destino final, es solo temporal si se usa
app.config['UPLOAD_FOLDER'] = TEMP_PROCESSING_FOLDER

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-fallback") # Use a strong, unique key in .env
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") # Change this for production!

# DB_PATH ahora estará en el disco persistente
DB_PATH = os.path.join(TEMP_PROCESSING_FOLDER, "logs/interactions.db")
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
# Helper Functions (These are now mostly used by Celery task)
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
    # This function is now logically part of the Celery worker's scope
    # but kept here if other parts of app.py were to use it synchronously.
    # For a clean separation, this function should be moved to celery_worker.py or a shared utility.
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
    # This function is now logically part of the Celery worker's scope
    # For a clean separation, this function should be moved to celery_worker.py or a shared utility.
    import mediapipe as mp # Import here to avoid top-level dependency if not used synchronously
    import cv2 # Import here
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
# Auth Routes
# ----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["password"] == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect("/admin")
        return "Contraseña incorrecta", 403
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ----------------------
# User Session Routes
# ----------------------

@app.route("/select", methods=["POST"])
def select():
    name = request.form["name"]
    email = request.form["email"]
    token = request.form["token"]
    today = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT active, start_date, end_date, token FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[0]:
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[1] <= today <= row[2]):
        return "Acceso fuera de rango permitido.", 403
    if row[3] != token:
        return "Token inválido. Verifica con RH.", 403

    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    conn = sqlite3.connect(DB_PATH) # Re-open connection for this query
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    if used_seconds >= 1800: # 30 minutes limit
        return "Has alcanzado el límite mensual de uso (30 minutos).", 403

    # Store user info in session for subsequent requests (chat, dashboard)
    session["name"] = name
    session["email"] = email
    session["token"] = token # Store token for dashboard validation

    return render_template("selector.html", name=name, email=email)

@app.route("/chat", methods=["POST"])
def chat():
    # Retrieve user info from session or form
    name = request.form.get("name") or session.get("name")
    email = request.form.get("email") or session.get("email")
    scenario = request.form["scenario"] if "scenario" in request.form else session.get("scenario")

    if not name or not email or not scenario:
        # If any essential data is missing, redirect to index or login
        return redirect(url_for('index'))

    # Update session with current scenario if it came from form
    session["scenario"] = scenario

    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    return render_template("chat.html", name=name, email=email, scenario=scenario, used_seconds=used_seconds)

@app.route("/dashboard", methods=["GET", "POST"]) # Allow GET for direct access from end_session
def dashboard():
    name = request.form.get("name") or session.get("name")
    email = request.form.get("email") or session.get("email")
    token = request.form.get("token") or session.get("token") # Try form first, then session
    today = date.today().isoformat()

    if not name or not email or not token:
        return redirect(url_for('index')) # Redirect if session data is missing

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT start_date, end_date, active, token FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close() # Close connection after fetch

    if not row:
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[2]: # active status
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[0] <= today <= row[1]):
        return "Acceso fuera de rango permitido.", 403
    if row[3] != token:
        return "Token inválido. Verifica con RH.", 403

    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    conn = sqlite3.connect(DB_PATH) # Re-open for this query
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    max_seconds = 1800 # 30 minutes limit

    # Re-fetch records using the proper columns (including 'tip' and 'visual_feedback')
    c.execute("SELECT scenario, message, evaluation, audio_path, timestamp, tip, visual_feedback FROM interactions WHERE name=? AND email=? ORDER BY timestamp DESC", (name, email))
    records = c.fetchall()
    conn.close()

    if used_seconds >= max_seconds and request.method == "POST": # Only block POST if limit reached
        # If coming from chat, it might already be at limit, allow dashboard view but no new sessions
        pass # Allow viewing dashboard even if limit is reached. The 'chat' route will block new sessions.

    return render_template("dashboard.html", name=name, email=email, records=records, used_seconds=used_seconds, max_seconds=max_seconds)

@app.route("/video/<path:filename>") # Cambiar a path para manejar URLs completas si es necesario
def serve_video(filename):
    # Opción 1: Redirigir directamente a la URL de S3 (más eficiente)
    s3_video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{filename}"
    print(f"[SERVE VIDEO] Redirigiendo a S3: {s3_video_url}")
    return redirect(s3_video_url, code=302) # Redirección temporal

@app.route("/upload_video", methods=["POST"])
def upload_video():
    name = request.form.get("name")
    email = request.form.get("email")
    file = request.files.get("video")

    if not file:
        return jsonify({"error": "No se envió ningún archivo de video."}), 400

    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    # Create a unique filename for the webm
    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    webm_filename = secure_filename(f"{name}_{email}_{timestamp_str}.webm")
    webm_filepath = os.path.join(app.config['UPLOAD_FOLDER'], webm_filename)
    
    try:
        file.save(webm_filepath)
        print(f"[UPLOAD] Video WEBM guardado en: {webm_filepath}")

        # --- NEW CODE ADDED HERE: Upload the WEBM file to S3 ---
        s3_url = upload_file_to_s3(webm_filepath, AWS_S3_BUCKET_NAME, webm_filename)
        if not s3_url:
            # If S3 upload fails, return an error to the frontend
            print(f"[ERROR] Falló la subida del WEBM original a S3: {webm_filepath}")
            return jsonify({"status": "error", "message": "Failed to upload video to S3."}), 500
        print(f"[S3 UPLOAD] Video WEBM subido a S3: {s3_url}")
        # --- END NEW CODE ---

        # Store the filename in session to link with log_full_session later
        # This filename (webm_filename) is what the Celery worker will use to look for in S3
        session["last_video_filename"] = webm_filename

        # Optionally, update last interaction to include video path if an entry already exists
        # This might be redundant if log_full_session is always called after upload_video
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Attempt to find a recent interaction for this user/scenario to update it
        # This assumes 'upload_video' happens very close to an interaction start.
        # A more robust solution might pass an an interaction_id from frontend.
        c.execute("SELECT id FROM interactions WHERE name = ? AND email = ? ORDER BY timestamp DESC LIMIT 1", (name, email))
        row = c.fetchone()
        if row:
            # Update audio_path to point to the S3 URL of the original WEBM.
            # The Celery task will later update it to the S3 URL of the converted MP4.
            c.execute("UPDATE interactions SET audio_path = ? WHERE id = ?", (s3_url, row[0]))
            conn.commit()
        conn.close()

        # Return the S3 URL of the uploaded WEBM file to the frontend
        return jsonify({"status": "saved", "path": s3_url})

    except Exception as e:
        print(f"[ERROR] Error during video upload process: {e}")
        return jsonify({"error": f"Error al procesar la subida del video: {str(e)}"}), 500

@app.route("/log_full_session", methods=["POST"])
def log_full_session():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    duration = int(data.get("duration", 0))
    video_filename = data.get("video_filename") # This is the original .webm filename

    # Store necessary data for the Celery task
    task_data = {
        "name": name,
        "email": email,
        "scenario": scenario,
        "conversation": conversation,
        "duration": duration,
        "video_object_key": video_filename # This should be the filename in S3 for the WEBM
    }

    # Dispatch the processing to a Celery task asynchronously
    task = celery_app.send_task('celery_worker.process_session_video', args=[task_data]) # Explicitly sending by name

    print(f"[CELERY] Task dispatched: {task.id} for user {email}")

    # Return immediate response to frontend
    return jsonify({
        "status": "processing",
        "message": "Tu sesión está siendo analizada. Puedes ver el progreso en tu Dashboard en unos minutos.",
        "task_id": task.id
    })

# ----------------------
# Admin Routes
# ----------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    print(f"DEBUG: Accediendo a /admin. Método HTTP: {request.method}")

    if not session.get("admin"):
        print("DEBUG: No hay sesión de administrador, redirigiendo a /login")
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # POST: user management
    if request.method == "POST":
        print("DEBUG: Recibida solicitud POST en /admin")
        action = request.form.get("action")
        print(f"DEBUG: Acción del formulario: {action}")
        if action == "add":
            print("DEBUG: Intentando añadir usuario")
            name = request.form["name"]
            email = request.form["email"]
            start = request.form["start_date"]
            end = request.form["end_date"] # This was the missing name attribute fix
            token = secrets.token_hex(8) # Generate a new token
            print(f"DEBUG: Datos de usuario: {name}, {email}, {start}, {end}, {token}")
            try:
                c.execute("""INSERT OR REPLACE INTO users (name, email, start_date, end_date, active, token)
                                   VALUES (?, ?, ?, ?, 1, ?)""", (name, email, start, end, token))
                conn.commit()
                print(f"[ADMIN] Added/Updated user: {email}")
            except Exception as e:
                print(f"ERROR: Falló al insertar usuario en DB: {e}")
                conn.rollback()
                return f"Error al guardar usuario: {str(e)}", 500
        elif action == "toggle":
            print("DEBUG: Intentando activar/desactivar usuario")
            user_id = int(request.form["user_id"])
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (user_id,))
            print(f"[ADMIN] Toggled user active status: {user_id}")
        elif action == "regen_token":
            print("DEBUG: Intentando regenerar token")
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

# The if __name__ == "__main__": block has been removed for deployment with Gunicorn.
# Gunicorn is responsible for starting the application.