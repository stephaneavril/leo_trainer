FROM python:3.11-slim

# Instala ffmpeg y git
RUN apt-get update && apt-get install -y ffmpeg git

# Copia los archivos del proyecto
WORKDIR /app
COPY . /app

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto de Flask
EXPOSE 10000

# Comando para iniciar tu app
CMD ["python", "app.py"]
