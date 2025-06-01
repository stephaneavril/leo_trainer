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

print("\U0001F680 Iniciando Leo Virtual Trainer...")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai # This correctly sets the OpenAI client

# Load Whisper model once when the app starts
try:
    whisper_model = whisper.load_model("base")
    print("\U0001F3A7 Whisper model loaded successfully.")
except Exception as e:
    print(f"\U0001F525 Error loading Whisper model: {e}")
    whisper_model = None # Handle case where model fails to load

# --- Configuration ---
AUDIO_FOLDER = "/var/data/" # Use the actual mount path + a subdirectory
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = AUDIO_FOLDER

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-fallback") # Use a strong, unique key in .env
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") # Change this for production!

DB_PATH = "logs/interactions.db"
os.makedirs("logs", exist_ok=True)

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
        audio_path TEXT,        -- Path to recorded audio/video file
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
@app.route("/")
def index():
    return render_template("index.html")

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

    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
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

@app.route("/video/<filename>")
def serve_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return f"Archivo no encontrado: {filepath}", 404
    print(f"Serving file: {filepath}")
    return send_file(filepath)

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
    webm_filename = secure_filename(f"{name}_{email}_{timestamp_str}.webm")
    webm_filepath = os.path.join(app.config['UPLOAD_FOLDER'], webm_filename)
    file.save(webm_filepath)
    print(f"[UPLOAD] Video WEBM guardado en: {webm_filepath}")

    session["last_video_filename"] = webm_filename # Store just the filename

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM interactions WHERE name = ? AND email = ? ORDER BY timestamp DESC LIMIT 1", (name, email))
    row = c.fetchone()
    if row:
        c.execute("UPDATE interactions SET audio_path = ? WHERE id = ?", (webm_filename, row[0])) # Store just the filename
        conn.commit()
    conn.close()

    # Return only the filename, not the full path
    return jsonify({"status": "saved", "path": webm_filename})

@app.route("/log_full_session", methods=["POST"])
def log_full_session():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    duration = int(data.get("duration", 0))
    video_filename = data.get("video_filename") # This is the original .webm filename

    timestamp = datetime.now().isoformat()
    transcribed_text = ""
    public_summary = "Evaluación no disponible."
    internal_summary = "Evaluación no disponible."
    tip_text = "Consejo no disponible."
    posture_feedback = "Análisis visual no realizado."

    if not video_filename:
        print("[ERROR] log_full_session: No se recibió video_filename.")
        return jsonify({"status": "error", "error": "No se recibió el nombre del archivo de video."}), 400

    original_webm_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    mp4_filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_filename.replace('.webm', '.mp4'))
    temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_from_{video_filename.split('.')[0]}.wav")

    # --- VIDEO CONVERSION AND AUDIO EXTRACTION ---
    if not os.path.exists(original_webm_path):
        print(f"[ERROR] Original WEBM video file not found: {original_webm_path}")
        return jsonify({"status": "error", "error": f"Archivo de video WEBM no encontrado: {original_webm_path}"}), 400

    print(f"[INFO] Processing video: {original_webm_path}")
    if convert_webm_to_mp4(original_webm_path, mp4_filepath):
        video_to_process_path = mp4_filepath
        print(f"[INFO] Converted to MP4: {mp4_filepath}")
    else:
        video_to_process_path = original_webm_path # Try processing original if conversion fails
        print(f"[WARNING] Failed to convert to MP4, attempting to process original WEBM: {original_webm_path}")

    # Initialize video_clip and cap to None outside try blocks
    # so they can be accessed in finally blocks, even if initial assignments fail.
    video_clip = None
    cap = None

    # --- TRANSCRIPTION (from video's audio) ---
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
            if video_clip: # Ensure the clip is closed if it was opened
                video_clip.close() #
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    else:
        print("[WARNING] Whisper model not loaded, skipping audio transcription.")

    # --- Prepare text for GPT evaluation ---
    # Use the transcribed text primarily for the user's side if available,
    # otherwise fall back to frontend conversation data.
    # Note: D-ID agent might already provide accurate transcripts.
    user_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'user'])
    # Prioritize transcribed_text for user input if available and substantial
    final_user_text = transcribed_text if transcribed_text.strip() else user_dialogue

    leo_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'agent']) # D-ID agent usually uses 'agent' role
    if not leo_dialogue: # Fallback if agent uses 'leo' role
        leo_dialogue = " ".join([m['text'] for m in conversation if m['role'] == 'leo'])

    # Full conversation text for DB
    full_conversation_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in conversation])

    print(f"[DEBUG] Final USER_TEXT for AI: {final_user_text[:100]}...")
    print(f"[DEBUG] Final LEO_TEXT for AI: {leo_dialogue[:100]}...")

    if not final_user_text.strip() and not leo_dialogue.strip():
        print("[ERROR] No hay texto de usuario ni de Leo para evaluar. Verifica que el video tenga audio y que el frontend envíe la conversación.")
        return jsonify({
            "status": "error",
            "evaluation": "No se encontró contenido válido para evaluar.",
            "tip": "Asegúrate de hablar y de que el asistente Leo también responda durante la sesión."
        }), 400

    # --- AI EVALUATION (GPT) ---
    try:
        summaries = evaluate_interaction(final_user_text, leo_dialogue) # Use the specific texts
        public_summary = summaries.get("public", public_summary)
        internal_summary = summaries.get("internal", internal_summary)
        print(f"[INFO] AI Evaluation successful. Public: {public_summary[:50]}...")
    except Exception as e:
        public_summary = "⚠️ Evaluación automática no disponible."
        internal_summary = f"❌ Error en evaluación: {str(e)}"
        print(f"[ERROR] Error calling evaluate_interaction: {e}")

    # --- PERSONALIZED TIP (GPT) ---
    try:
        tip_completion = client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for potentially better performance/cost
            messages=[
                {"role": "system", "content": "Eres un coach médico empático y útil. Ofrece 2-3 consejos prácticos, claros y concretos sobre cómo mejorar las interacciones de un representante médico con doctores. Enfócate en el participante."},
                {"role": "user", "content": f"Basado en esta conversación:\n\nParticipante: {final_user_text}\nLeo: {leo_dialogue}\n\n¿Qué podría hacer mejor el participante la próxima vez? Ofrece consejos accionables y positivos."}
            ],
            temperature=0.7, # Slightly higher temperature for more creative advice
        )
        tip_text = tip_completion.choices[0].message.content.strip()
        print(f"[INFO] AI Tip generated: {tip_text[:50]}...")
    except Exception as e:
        tip_text = f"⚠️ No se pudo generar un consejo automático: {str(e)}"
        print(f"[ERROR] Error generating personalized tip: {e}")

    # --- VISUAL ANALYSIS ---
    if video_to_process_path and os.path.exists(video_to_process_path):
        try:
            print(f"[VIDEO ANALYSIS] Starting posture analysis for: {video_to_process_path}")
            # The analyze_video_posture function now explicitly handles cap.release() internally,
            # so we just call it.
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
        print(f"[WARNING] Skipping visual analysis: Video file not found at {video_to_process_path}")
        posture_feedback = "Video no disponible para análisis visual."

    # --- SAVE TO DATABASE ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""INSERT INTO interactions (
            name, email, scenario, message, response, timestamp,
            evaluation, evaluation_rh, duration_seconds, tip,
            audio_path, visual_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            name, email, scenario, full_conversation_text, leo_dialogue, timestamp,
            public_summary, internal_summary, duration, tip_text,
            video_filename, posture_feedback # Store the original webm filename
        ))
        conn.commit()
        print(f"[DB] Interaction saved for {email}, scenario {scenario}.")
    except Exception as e:
        print(f"[ERROR] Database insert failed: {e}")
        # Optionally, roll back if needed
        # conn.rollback()
        return jsonify({"status": "error", "error": f"Error al guardar en la base de datos: {str(e)}"}), 500
    finally:
        conn.close()

    # --- Cleanup temporary files ---
    try:
        if os.path.exists(mp4_filepath):
            os.remove(mp4_filepath)
            print(f"Cleaned up {mp4_filepath}")
        # Keep the original .webm for serving if needed, or remove it too
        # if os.path.exists(original_webm_path):
        #     os.remove(original_webm_path)
        #     print(f"Cleaned up {original_webm_path}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to remove temp files: {e}")

    # --- Return JSON Response to Frontend ---
    return jsonify({
        "status": "ok",
        "evaluation": public_summary,
        "tip": tip_text,
        "visual_feedback": posture_feedback # Send visual feedback to frontend too if needed
    })

# ----------------------
# Admin Routes
# ----------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if not session.get("admin"):
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # POST: user management
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            name = request.form["name"]
            email = request.form["email"]
            start = request.form["start_date"]
            end = request.form["end_date"]
            token = secrets.token_hex(8) # Generate a new token
            c.execute("""INSERT OR REPLACE INTO users (name, email, start_date, end_date, active, token)
                               VALUES (?, ?, ?, ?, 1, ?)""", (name, email, start, end, token))
            print(f"[ADMIN] Added/Updated user: {email}")
        elif action == "toggle":
            user_id = int(request.form["user_id"])
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (user_id,))
            print(f"[ADMIN] Toggled user active status: {user_id}")
        elif action == "regen_token":
            user_id = int(request.form["user_id"])
            new_token = secrets.token_hex(8)
            c.execute("UPDATE users SET token = ? WHERE id = ?", (new_id, user_id)) # Corrected 'user_id' to 'new_token'
            print(f"[ADMIN] Regenerated token for user: {user_id}")
        conn.commit()

    # Load interactions
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
        # 1. Get the video file path from the database
        c.execute("SELECT audio_path FROM interactions WHERE id = ?", (session_id,))
        row = c.fetchone()

        if row and row[0]: # Check if a record and audio_path exist
            video_filename = row[0]
            video_filepath = os.path.join(AUDIO_FOLDER, video_filename)

            # 2. Delete the physical video file
            if os.path.exists(video_filepath):
                os.remove(video_filepath)
                print(f"[DELETE] Successfully deleted video file: {video_filepath}")
            else:
                print(f"[DELETE WARNING] Video file not found at: {video_filepath} (record will still be deleted)")

            # 3. Delete the record from the database
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # In production, set debug=False and use a production WSGI server like Gunicorn
    app.run(host="0.0.0.0", port=port, debug=True)