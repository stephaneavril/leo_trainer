# app.py
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
        timestamp TEXT
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
    return render_template("selector.html", name=name, email=email)

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
    c.execute("SELECT name, email, scenario, message, response, audio_path, timestamp FROM interactions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
