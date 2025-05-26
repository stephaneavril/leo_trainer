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

print("🚀 Iniciando Leo Virtual Trainer...")

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
    today = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT active, start_date, end_date FROM users WHERE email=?", (email,))
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

    # Verificar tiempo restante del mes
    now = datetime.now()
    start_of_month = now.replace(day=1).isoformat()
    c.execute("""
        SELECT SUM(duration_seconds) FROM interactions
        WHERE email = ? AND timestamp >= ?
    """, (email, start_of_month))
    used_seconds = c.fetchone()[0] or 0
    conn.close()

    if used_seconds >= 600:
        return "Has alcanzado el límite mensual de uso (10 minutos).", 403

    return render_template("selector.html", name=name, email=email)

@app.route("/admin/users", methods=["GET", "POST"])
def manage_users():
    if not session.get("admin"):
        return redirect("/login")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        start = request.form["start_date"]
        end = request.form["end_date"]
        c.execute("INSERT OR REPLACE INTO users (name, email, start_date, end_date, active) VALUES (?, ?, ?, ?, 1)",
                  (name, email, start, end))
        conn.commit()
    c.execute("SELECT id, name, email, start_date, end_date, active FROM users")
    users = c.fetchall()
    conn.close()
    return render_template("users.html", users=users)

@app.route("/admin/users/deactivate/<int:user_id>")
def deactivate_user(user_id):
    if not session.get("admin"):
        return redirect("/login")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET active=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("manage_users"))

@app.route("/chat", methods=["POST"])
def chat():
    name = request.form["name"]
    email = request.form["email"]
    scenario = request.form["scenario"]
    return render_template("chat.html", name=name, email=email, scenario=scenario)

@app.route("/dashboard", methods=["POST"])
def dashboard():
    name = request.form["name"]
    email = request.form["email"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Obtener sesiones
    c.execute("""
        SELECT scenario, message, evaluation, audio_path, timestamp
        FROM interactions
        WHERE name=? AND email=?
        ORDER BY timestamp DESC
    """, (name, email))
    records = c.fetchall()

    # Obtener minutos usados
    c.execute("""
        SELECT SUM(duration_seconds)
        FROM interactions
        WHERE email = ? AND timestamp >= ?
    """, (email, datetime.now().replace(day=1).isoformat()))
    used_seconds = c.fetchone()[0] or 0
    max_seconds = 600

    conn.close()

    return render_template(
        "dashboard.html",
        name=name,
        email=email,
        records=records,
        used_seconds=used_seconds,
        max_seconds=max_seconds
    )

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

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, scenario, message, response, timestamp, evaluation, evaluation_rh, duration_seconds)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (name, email, scenario, full_text, leo_text, timestamp, public_summary, internal_summary, duration))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "evaluation": public_summary})

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
