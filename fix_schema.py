import sqlite3

DB_PATH = "logs/interactions.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Agregar columna si no existe
try:
    c.execute("ALTER TABLE interactions ADD COLUMN duration_seconds INTEGER DEFAULT 0")
    print("✅ Columna 'duration_seconds' agregada correctamente.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("ℹ️ La columna 'duration_seconds' ya existe.")
    else:
        raise

conn.commit()
conn.close()
