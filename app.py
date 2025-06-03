# app.py

import os
import json
import sqlite3
import secrets
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from celery import Celery
from celery_worker import celery_app 

import openai
import subprocess 

import boto3
from botocore.exceptions import ClientError

print("\U0001F680 Iniciando Leo Virtual Trainer...")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai 

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

PERSISTENT_DISK_MOUNT_PATH = os.getenv("PERSISTENT_DISK_MOUNT_PATH", "/var/data") 
TEMP_PROCESSING_FOLDER = os.path.join(PERSISTENT_DISK_MOUNT_PATH, "leo_trainer_processing") 
os.makedirs(TEMP_PROCESSING_FOLDER, exist_ok=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
CORS(app) 

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

app.config['UPLOAD_FOLDER'] = TEMP_PROCESSING_FOLDER

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-fallback") 
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") 

DB_PATH = os.path.join(TEMP_PROCESSING_FOLDER, "logs/interactions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

    c.execute("PRAGMA table_info(interactions)")
    columns = [col[1] for col in c.fetchall()]
    if 'evaluation_rh' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN evaluation_rh TEXT")
        print("Added 'evaluation_rh' to interactions table.")

    if 'tip' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN tip TEXT")
        print("Added 'tip' to interactions table.")

    if 'visual_feedback' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN visual_feedback TEXT")
        print("Added 'visual_feedback' to interactions table.")

    c.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in c.fetchall()]
    if 'token' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN token TEXT UNIQUE")
        print("Added 'token' to users table.")

    conn.commit()
    conn.close()
    print("\U0001F527 Database schema patched.")

init_db()
patch_db_schema()

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
    """Analyzes video for face detection as a proxy for posture/presence."""
    import mediapipe as mp 
    import cv2 
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
                    break 
                summary["frames_total"] += 1
                results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    summary["face_detected_frames"] += 1
        cap.release()
    except Exception as e:
        summary["error"] = str(e)
        print(f"[ERROR] Error during video posture analysis: {e}")
    return summary

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
    conn = sqlite3.connect(DB_PATH) 
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    if used_seconds >= 1800: 
        return "Has alcanzado el límite mensual de uso (30 minutos).", 403

    session["name"] = name
    session["email"] = email
    session["token"] = token 

    return render_template("selector.html", name=name, email=email)

@app.route("/chat", methods=["POST"])
def chat():
    # CAMBIO: Priorizar request.form para token, luego session
    name = request.form.get("name") or session.get("name")
    email = request.form.get("email") or session.get("email")
    scenario = request.form["scenario"] if "scenario" in request.form else session.get("scenario")
    token = request.form.get("token") or session.get("token") # Obtener el token del formulario o de la sesión

    if not name or not email or not scenario or not token: 
        print(f"DEBUG: Redirigiendo a index desde /chat. Datos faltantes: Name={name}, Email={email}, Scenario={scenario}, Token={token}")
        return redirect(url_for('index'))

    session["scenario"] = scenario

    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    return render_template("chat.html", name=name, email=email, scenario=scenario, used_seconds=used_seconds, token=token) 

@app.route("/dashboard", methods=["GET", "POST"]) 
def dashboard():
    name = request.form.get("name") or session.get("name")
    email = request.form.get("email") or session.get("email")
    token = request.form.get("token") or session.get("token") 
    print(f"DEBUG: Dashboard access - Name: {name}, Email: {email}, Token: {token}") 
    today = date.today().isoformat()

    if not name or not email or not token:
        print(f"DEBUG: Redirecting to index. Missing data: Name={name}, Email={email}, Token={token}") 
        return redirect(url_for('index')) 

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT start_date, end_date, active, token FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close() 

    if not row:
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[2]: 
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[0] <= today <= row[1]):
        return "Acceso fuera de rango permitido.", 403
    if row[3] != token:
        return "Token inválido. Verifica con RH.", 403

    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    conn = sqlite3.connect(DB_PATH) 
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    max_seconds = 1800 

    c.execute("SELECT scenario, message, evaluation, audio_path, timestamp, tip, visual_feedback FROM interactions WHERE name=? AND email=? ORDER BY timestamp DESC", (name, email))
    records = c.fetchall()
    conn.close()

    if used_seconds >= max_seconds and request.method == "POST": 
        pass 

    return render_template("dashboard.html", name=name, email=email, records=records, used_seconds=used_seconds, max_seconds=max_seconds, token=token) 

@app.route("/video/<path:filename>") 
def serve_video(filename):
    s3_video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{filename}"
    print(f"[SERVE VIDEO] Redirigiendo a S3: {s3_video_url}")
    return redirect(s3_video_url, code=302) 

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files.get('video')
    name = request.form.get('name')
    email = request.form.get('email')

    if not video_file or not name or not email:
        return jsonify({'status': 'error', 'message': 'Faltan datos (video, nombre o correo).'}), 400

    filename = secure_filename(f"{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm")
    local_path = os.path.join("temp_uploads", filename)
    os.makedirs("temp_uploads", exist_ok=True)
    video_file.save(local_path)

    try:
        s3_key = filename
        s3_url = upload_file_to_s3(local_path, AWS_S3_BUCKET_NAME, s3_key) 
        if not s3_url:
            raise Exception("No se pudo subir el archivo a S3.")
        
        print(f"[S3] Subido a: {s3_url}")

        session["last_video_s3_key"] = s3_key 
        print(f"DEBUG: Stored s3_key in session: {session['last_video_s3_key']}")


        return jsonify({'status': 'ok', 'video_url': s3_url, 's3_object_key': s3_key}) 
    except Exception as e:
        print(f"[ERROR] upload_video: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

@app.route("/log_full_session", methods=["POST"])
def log_full_session():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    duration = int(data.get("duration", 0))

    video_object_key = session.pop("last_video_s3_key", None) 
    
    if not video_object_key:
        print("[ERROR] log_full_session: No se encontró video_object_key en la sesión.")
        return jsonify({
            "status": "error",
            "message": "Error interno: No se pudo encontrar el video subido para procesar."
        }), 500

    task_data = {
        "name": name,
        "email": email,
        "scenario": scenario,
        "conversation": conversation,
        "duration": duration,
        "video_object_key": video_object_key 
    }

    from celery_worker import process_session_video
    task = process_session_video.delay(task_data)

    print(f"[CELERY] Task dispatched: {task.id} for user {email}")

    return jsonify({
        "status": "processing",
        "message": "Tu sesión está siendo analizada. Puedes ver el progreso en tu Dashboard en unos minutos.",
        "task_id": task.id
    })

@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    print(f"DEBUG: Accediendo a /admin. Método HTTP: {request.method}")

    if not session.get("admin"):
        print("DEBUG: No hay sesión de administrador, redirigiendo a /login")
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if request.method == "POST":
        print("DEBUG: Recibida solicitud POST en /admin")
        action = request.form.get("action")
        print(f"DEBUG: Acción del formulario: {action}")
        if action == "add":
            print("DEBUG: Intentando añadir usuario")
            name = request.form["name"]
            email = request.form["email"]
            start = request.form["start_date"]
            end = request.form["end_date"] 
            token = secrets.token_hex(8) 
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

    c.execute("""SELECT name, email, scenario, message, response, audio_path, timestamp, evaluation, evaluation_rh, tip, visual_feedback
                 FROM interactions
                 ORDER BY timestamp DESC""")
    raw_data = c.fetchall()

    processed_data = []
    for row in raw_data:
        try:
            parsed_rh_evaluation = json.loads(row[8])
        except (json.JSONDecodeError, TypeError):
            parsed_rh_evaluation = {"error": "Invalid JSON or old format", "raw_content": row[8]}

        processed_row = list(row)
        processed_row[8] = parsed_rh_evaluation 
        processed_data.append(processed_row)

    c.execute("SELECT id, name, email, start_date, end_date, active, token FROM users")
    users = c.fetchall()

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

    contracted_minutes = 1050 

    conn.close()

    return render_template(
        "admin.html",
        data=processed_data, 
        users=users,
        usage_summaries=usage_summaries,
        total_minutes=total_minutes_all_users,
        contracted_minutes=contracted_minutes
    )

@app.route("/end_session", methods=["POST", "GET"]) 
def end_session_redirect():
    name = session.get("name")
    email = session.get("email")
    if name and email:
        return redirect(url_for('dashboard', name=name, email=email), code=307 if request.method == "POST" else 302)
    return redirect(url_for('index'))

@app.route("/admin/delete_session/<int:session_id>", methods=["POST"])
def delete_session(session_id):
    if not session.get("admin"):
        return redirect("/login") 

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. Get the video URL from the database
        c.execute("SELECT audio_path FROM interactions WHERE id = ?", (session_id,))
        row = c.fetchone()

        if row and row[0]: 
            s3_video_url = row[0]
            s3_object_key = s3_video_url.split('/')[-1]

            try:
                s3_client.delete_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_object_key)
                print(f"[DELETE S3] Successfully deleted S3 object: s3://{AWS_S3_BUCKET_NAME}/{s3_object_key}")
            except ClientError as e:
                print(f"[DELETE S3 ERROR] Failed to delete S3 object {s3_object_key}: {e}")
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"[DELETE S3 WARNING] S3 object not found at: {s3_object_key} (record will still be deleted)")
                else:
                    return f"Error al eliminar el video de S3: {str(e)}", 500

            c.execute("DELETE FROM interactions WHERE id = ?", (session_id,))
            conn.commit()
            print(f"[DB DELETE] Successfully deleted record for session ID: {session_id}")
            return redirect("/admin") 
        else:
            print(f"[DELETE ERROR] Session or video path not found for ID: {session_id}")
            return "Sesión no encontrada o sin video asociado.", 404

    except Exception as e:
        conn.rollback() 
        print(f"[DELETE ERROR] Failed to delete session {session_id}: {e}")
        return f"Error al eliminar la sesión: {str(e)}", 500
    finally:
        conn.close()

@app.route("/healthz")
def health_check():
    return "OK", 200
