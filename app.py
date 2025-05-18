# app.py (solo para registrar conversación de agente Leo en tiempo real)
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import openai

load_dotenv()

app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv("OPENAI_API_KEY")

DB_PATH = "logs/interactions.db"


def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        message TEXT,
        response TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/log", methods=["POST"])
def log_interaction():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    response = data.get("response")
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO interactions (name, email, message, response, timestamp)
                 VALUES (?, ?, ?, ?, ?)""",
              (name, email, message, response, timestamp))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

