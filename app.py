# app.py (con grabación de audio y flujo completo)
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/audios'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = "logs/interactions.db"

def init_db():
    os.makedirs("logs", exist_ok=True)
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
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

@app.route("/")
def login():
    return render_template("index.html")

@app.route("/select", methods=["POST"])
def select():
    name = request.form.get("name")
    email = request.form.get("email")
    return render_template("selector.html", name=name, email=email)

@app.route("/chat", methods=["POST"])
def chat():
    name = request.form.get("name")
    email = request.form.get("email")
    scenario = request.form.get("scenario")
    return render_template("chat.html", name=name, email=email, scenario=scenario)

@app.route("/log", methods=["POST"])
def log_interaction():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    response = data.get("response")
    scenario = data.get("scenario")
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, scenario, message, response, audio_path, timestamp)
                 VALUES (?, ?, ?, ?, ?, '', ?)""",
              (name, email, scenario, message, response, timestamp))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    name = request.form.get("name")
    email = request.form.get("email")
    file = request.files['audio']
    filename = secure_filename(f"{name}_{datetime.now().isoformat()}.webm")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Actualiza el registro más reciente del usuario con el audio
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""UPDATE interactions SET audio_path=? 
                 WHERE name=? AND email=? 
                 ORDER BY id DESC LIMIT 1""",
              (filepath, name, email))
    conn.commit()
    conn.close()
    return jsonify({"status": "audio saved"})

@app.route("/admin")
def admin_panel():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, email, scenario, message, response, audio_path, timestamp FROM interactions ORDER BY timestamp DESC")
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))