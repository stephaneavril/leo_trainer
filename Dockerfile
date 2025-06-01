# Usa una imagen base de Python liviana
FROM python:3.11-slim

# Instala ffmpeg y otras dependencias básicas
RUN apt-get update && apt-get install -y ffmpeg

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto que usa tu app (ajusta si es diferente)
EXPOSE 10000

# Comando para iniciar tu app
CMD ["python", "app.py"]
