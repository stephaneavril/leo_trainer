import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Cargar variables de entorno
load_dotenv()

# Configurar Flask
app = Flask(__name__)
CORS(app)

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ruta a base de datos
DB_PATH = "logs/interactions.db"

# Inicializar base de datos
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

# Ruta principal
@app.route("/")
def index():
    return render_template("chat.html")

# Ruta para guardar interacciones
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

# Panel de administración para visualizar conversaciones
@app.route("/admin")
def admin_panel():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, email, message, response, timestamp FROM interactions ORDER BY timestamp DESC")
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)

# Iniciar servidor
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
