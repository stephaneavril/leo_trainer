# app.py
import os
import sqlite3
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import evaluate_interaction

print("\U0001F680 Iniciando Leo Virtual Trainer...")

load_dotenv()

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

DB_PATH = "logs/interactions.db"
UPLOAD_FOLDER = "static/audios"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------
# DB Init
# ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        scenario TEXT,
        message TEXT,
        response TEXT,
        audio_path TEXT,
        timestamp TEXT,
        evaluation TEXT,
        evaluation_rh TEXT,
        duration_seconds INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        start_date TEXT,
        end_date TEXT,
        active INTEGER DEFAULT 1
    )''')
    conn.commit()
    conn.close()

def patch_db_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Verifica y agrega 'evaluation_rh' si no existe
    c.execute("PRAGMA table_info(interactions)")
    columns = [col[1] for col in c.fetchall()]
    if 'evaluation_rh' not in columns:
        c.execute("ALTER TABLE interactions ADD COLUMN evaluation_rh TEXT")

    # Verifica y agrega 'token' si no existe
    c.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in c.fetchall()]
    if 'token' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN token TEXT")

    conn.commit()
    conn.close()

init_db()
patch_db_schema()

# ----------------------
# Auth
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
# Routes
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

    if not row:
        conn.close()
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[0]:
        conn.close()
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[1] <= today <= row[2]):
        conn.close()
        return "Acceso fuera de rango permitido.", 403
    if row[3] != token:
        conn.close()
        return "Token inválido. Verifica con RH.", 403

    start_of_month = datetime.now().replace(day=1).isoformat()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    if used_seconds >= 1800:
        return "Has alcanzado el límite mensual de uso (30 minutos).", 403

    return render_template("selector.html", name=name, email=email)

@app.route("/chat", methods=["POST"])
def chat():
    name = request.form["name"]
    email = request.form["email"]
    scenario = request.form["scenario"]

    # Guardar en la sesión
    session["name"] = name
    session["email"] = email
    session["scenario"] = scenario

    now = datetime.now()
    start_of_month = now.replace(day=1).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    return render_template("chat.html", name=name, email=email, scenario=scenario, used_seconds=used_seconds)

@app.route("/dashboard", methods=["POST"])
def dashboard():
    name = request.form["name"]
    email = request.form["email"]
    token = request.form.get("token") or session.get("token")
    today = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT start_date, end_date, active, token FROM users WHERE email = ?", (email,))
    row = c.fetchone()

    if not row:
        conn.close()
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[2]:
        conn.close()
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[0] <= today <= row[1]):
        conn.close()
        return "Acceso fuera de rango permitido.", 403
    if row[3] != token:
        conn.close()
        return "Token inválido. Verifica con RH.", 403

    # ✅ Guardar sesión tras validación
    session["name"] = name
    session["email"] = email
    session["token"] = token

    now = datetime.now()
    start_of_month = now.replace(day=1).isoformat()
    c.execute("SELECT SUM(duration_seconds) FROM interactions WHERE email = ? AND timestamp >= ?", (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    max_seconds = 1800

    c.execute("SELECT scenario, message, evaluation, audio_path, timestamp FROM interactions WHERE name=? AND email=? ORDER BY timestamp DESC", (name, email))
    records = c.fetchall()
    conn.close()

    if used_seconds >= max_seconds:
        return "Has alcanzado el límite mensual de uso (30 minutos).", 403

    return render_template("dashboard.html", name=name, email=email, records=records, used_seconds=used_seconds, max_seconds=max_seconds)

@app.route("/video/<filename>")
def serve_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(filepath):
        return f"Archivo no encontrado: {filepath}", 404
    return send_file(filepath)

@app.route("/upload_video", methods=["POST"])
def upload_video():
    name = request.form.get("name")
    email = request.form.get("email")
    file = request.files.get("video")

    if not file:
        return "No video uploaded", 400

    filename = secure_filename(f"{name}_{email}_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM interactions WHERE name = ? AND email = ? ORDER BY timestamp DESC LIMIT 1", (name, email))
    row = c.fetchone()
    if row:
        c.execute("UPDATE interactions SET audio_path = ? WHERE id = ?", (filename, row[0]))
        conn.commit()
    conn.close()
    return jsonify({"status": "saved", "path": filename})

@app.route("/log_full_session", methods=["POST"])
def log_full_session():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    timestamp = datetime.now().isoformat()
    duration = int(data.get("duration", 0))

    full_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in conversation])
    user_text = " ".join([m['text'] for m in conversation if m['role'] == 'user'])
    leo_text = " ".join([m['text'] for m in conversation if m['role'] == 'leo'])

    try:
        summaries = evaluate_interaction(user_text, leo_text)
        public_summary = summaries.get("public", "")
        internal_summary = summaries.get("internal", "")
    except Exception as e:
        public_summary = "⚠️ Evaluación no disponible."
        internal_summary = f"❌ Error: {str(e)}"

    # Generar consejo personalizado post-sesión (con GPT)
    try:
        tip_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un coach médico que ofrece consejos prácticos sobre cómo mejorar las interacciones con doctores."},
                {"role": "user", "content": f"""Basado en esta conversación:

{user_text}

¿Qué podría hacer mejor el participante la próxima vez? Da 2-3 sugerencias claras y concretas."""}
            ],
            temperature=0.5,
        )
        tip_text = tip_completion.choices[0].message.content.strip()
    except Exception as e:
        tip_text = f"⚠️ No se pudo generar consejo automático: {str(e)}"

    # Guardar interacción
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, scenario, message, response, timestamp, evaluation, evaluation_rh, duration_seconds)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (name, email, scenario, full_text, leo_text, timestamp, public_summary, internal_summary, duration))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "evaluation": public_summary, "tip": tip_text})

import secrets  # fuera de las funciones


@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if not session.get("admin"):
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # POST: gestión de usuarios
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            name = request.form["name"]
            email = request.form["email"]
            start = request.form["start_date"]
            end = request.form["end_date"]
            token = secrets.token_hex(8)
            c.execute("""INSERT OR REPLACE INTO users (name, email, start_date, end_date, active, token)
                         VALUES (?, ?, ?, ?, 1, ?)""", (name, email, start, end, token))
        elif action == "toggle":
            user_id = int(request.form["user_id"])
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (user_id,))
        elif action == "regen_token":
            user_id = int(request.form["user_id"])
            new_token = secrets.token_hex(8)
            c.execute("UPDATE users SET token = ? WHERE id = ?", (new_token, user_id))
        conn.commit()

    # Cargar interacciones
    c.execute("""SELECT name, email, scenario, message, response, audio_path, timestamp, evaluation, evaluation_rh
                 FROM interactions
                 ORDER BY timestamp DESC""")
    data = c.fetchall()

    # Cargar usuarios
    c.execute("SELECT id, name, email, start_date, end_date, active, token FROM users")
    users = c.fetchall()

    # Calcular minutos usados por usuario
    c.execute("""
        SELECT u.name, u.email, COALESCE(SUM(i.duration_seconds), 0) as used_secs
        FROM users u
        LEFT JOIN interactions i ON u.email = i.email
        GROUP BY u.name, u.email
    """)
    usage_rows = c.fetchall()

    usage_summaries = []
    total_minutes = 0
    for name, email, secs in usage_rows:
        mins = secs // 60
        total_minutes += mins
        summary = "Buen desempeño general" if mins >= 15 else "Actividad moderada" if mins >= 5 else "Poca actividad, se sugiere seguimiento"
        usage_summaries.append({
            "name": name,
            "email": email,
            "minutes": mins,
            "summary": summary
        })

    # Tiempo total contratado por RH (independiente del número de usuarios)
    contracted_minutes = 1050  # fijo: minutos contratados totales por el admin

    conn.close()

    return render_template(
        "admin.html",
        data=data,
        users=users,
        usage_summaries=usage_summaries,
        total_minutes=total_minutes,
        contracted_minutes=contracted_minutes
    )
@app.route("/end_session")
def end_session():
    name = session.get("name")
    email = session.get("email")
    scenario = session.get("scenario", "Desconocido")

    if not name or not email:
        return redirect("/")

    duration = 300  # 5 minutos en segundos
    timestamp = datetime.now().isoformat()

    # Guardar en la tabla `interactions` con duración vacía si no hay audio
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO interactions (name, email, scenario, message, response, duration_seconds, timestamp)
        VALUES (?, ?, ?, '', '', ?, ?)
    """, (name, email, scenario, duration, timestamp))
    conn.commit()
    conn.close()

    return redirect("/dashboard")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
