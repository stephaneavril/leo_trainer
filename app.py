# app.py
import os
import sqlite3
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import evaluate_interaction

load_dotenv()

app = Flask(__name__)
CORS(app)

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
        evaluation_rh TEXT
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
    conn.close()

    if not row:
        return "Usuario no registrado. Contacta a RH.", 403
    if not row[0]:
        return "Usuario inactivo. Contacta a RH.", 403
    if not (row[1] <= today <= row[2]):
        return "Acceso fuera de rango permitido.", 403

    return render_template("selector.html", name=name, email=email)

@app.route("/admin/users", methods=["GET", "POST"])
def manage_users():
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

@app.route("/log", methods=["POST"])
def log_interaction():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    response = data.get("response")
    scenario = data.get("scenario")
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, scenario, message, response, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (name, email, scenario, message, response, timestamp))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route("/log_full_session", methods=["POST"])
def log_full_session():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    scenario = data.get("scenario")
    conversation = data.get("conversation", [])
    timestamp = datetime.now().isoformat()

    full_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in conversation])
    user_text = next((m['text'] for m in reversed(conversation) if m['role'] == 'user'), "")
    leo_text = next((m['text'] for m in reversed(conversation) if m['role'] == 'leo'), "")
    summaries = evaluate_interaction(user_text, leo_text)
    public_summary = summaries.get("public", "")
    internal_summary = summaries.get("internal", "")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, scenario, message, response, timestamp, evaluation, evaluation_rh)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (name, email, scenario, full_text, leo_text, timestamp, public_summary, internal_summary))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "evaluation": public_summary})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    name = request.form.get("name")
    email = request.form.get("email")
    video = request.files.get("video")

    if not video:
        return "No video uploaded", 400

    filename = secure_filename(f"{name}_{datetime.now().isoformat().replace(':', '-')}.webm")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(save_path)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE interactions SET audio_path=? WHERE name=? AND email=? ORDER BY id DESC LIMIT 1",
              (save_path, name, email))
    conn.commit()
    conn.close()

    return jsonify({"status": "video saved"})

@app.route("/dashboard", methods=["POST"])
def dashboard():
    name = request.form["name"]
    email = request.form["email"]
    return render_template("dashboard.html", name=name, email=email)

@app.route("/admin")
def admin_panel():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, email, scenario, message, response, audio_path, timestamp, evaluation, evaluation_rh FROM interactions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)

@app.route("/admin/summary")
def admin_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, scenario, evaluation FROM interactions", conn)
    conn.close()

    user_counts = df['name'].value_counts()
    plt.figure(figsize=(10,6))
    user_counts.plot(kind='bar', color='skyblue')
    plt.title('Número de sesiones por usuario')
    plt.xlabel('Usuario')
    plt.ylabel('Cantidad de sesiones')
    plt.tight_layout()
    user_plot_path = "static/summary_plot.png"
    plt.savefig(user_plot_path)
    plt.close()

    kpi_terms = ["claridad", "objeci", "modelo de ventas", "cierre"]
    kpi_data = {term: [] for term in kpi_terms}
    scenarios = df['scenario'].unique()

    for scenario in scenarios:
        evals = df[df['scenario'] == scenario]['evaluation'].str.lower().fillna("")
        for term in kpi_terms:
            kpi_data[term].append(evals.str.contains(term).sum())

    kpi_df = pd.DataFrame(kpi_data, index=scenarios)
    kpi_df.plot(kind="bar", figsize=(12,6), colormap="coolwarm")
    plt.title("Métricas clave por escenario")
    plt.ylabel("# de sesiones con la métrica")
    plt.xlabel("Escenario")
    plt.xticks(rotation=45)
    plt.tight_layout()
    kpi_plot_path = "static/kpi_por_escenario.png"
    plt.savefig(kpi_plot_path)
    plt.close()

    return render_template("summary.html", plot_url=user_plot_path, kpi_url=kpi_plot_path)

@app.route("/export")
def export_excel():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, email, scenario, message, response, audio_path, timestamp, evaluation, evaluation_rh FROM interactions ORDER BY id DESC", conn)
    conn.close()
    export_path = "logs/leo_sessions.xlsx"
    df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True)

@app.route("/dashboard", methods=["POST"])
def dashboard():
    name = request.form["name"]
    email = request.form["email"]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT scenario, message, evaluation, audio_path, timestamp
        FROM interactions
        WHERE name=? AND email=?
        ORDER BY timestamp DESC
    """, (name, email))
    records = c.fetchall()
    conn.close()
    return render_template("dashboard.html", name=name, email=email, records=records)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
