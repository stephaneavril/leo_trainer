import sqlite3

DB_PATH = "logs/interactions.db"

required_columns = {
    "message": "TEXT",
    "response": "TEXT",
    "audio_path": "TEXT",
    "timestamp": "TEXT",
    "evaluation": "TEXT"
}

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Obtener las columnas actuales
c.execute("PRAGMA table_info(interactions)")
existing = [col[1] for col in c.fetchall()]

# Añadir las que faltan
for column, coltype in required_columns.items():
    if column not in existing:
        try:
            c.execute(f"ALTER TABLE interactions ADD COLUMN {column} {coltype}")
            print(f"✅ Columna añadida: {column}")
        except sqlite3.OperationalError as e:
            print(f"❌ Error añadiendo {column}: {e}")
    else:
        print(f"✔️ Columna ya existe: {column}")

conn.commit()
conn.close()
print("✅ Esquema de la base de datos actualizado.")
