FROM python:3.11-slim

# Instala git y ffmpeg, y otras dependencias del sistema
RUN apt-get update && \
    apt-get install -y git ffmpeg build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Establece directorio de trabajo
WORKDIR /app

# Copia solo lo necesario primero (para cacheo)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto del código
COPY . .

# Exponer el puerto de Flask
EXPOSE 10000

# Iniciar la aplicación
CMD ["python", "app.py"]
