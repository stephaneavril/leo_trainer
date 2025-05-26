# evaluator.py

import os
import re
import textwrap
import cv2
import numpy as np
from openai import OpenAI
from openai import OpenAIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_interaction(user_text, leo_text, video_path=None):
    """
    Evalúa la conversación y el lenguaje corporal del participante si hay video.
    Retorna evaluación pública (usuario) y evaluación RH (técnica).
    """

    def basic_keywords_eval(text):
        score = 0
        keywords = ["beneficio", "estudio", "síntoma", "tratamiento", "reflujo", "mecanismo", "eficacia", "seguridad"]
        for kw in keywords:
            if kw in text.lower():
                score += 1
        return score

    def detect_closure_language(text):
        closure_patterns = ["compromiso", "siguiente paso", "acordamos", "puedo contar con"]
        return any(p in text.lower() for p in closure_patterns)

    def detect_visual_cues_from_video(path):
        try:
            cap = cv2.VideoCapture(path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frontal_frames = 0
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            for _ in range(min(100, frame_count)):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    frontal_frames += 1
            cap.release()
            ratio = frontal_frames / min(100, frame_count)
            if ratio >= 0.7:
                return "✅ Te mostraste visible y profesional frente a cámara.", "Correcta"
            else:
                return "⚠️ Asegúrate de mantenerte visible durante toda la sesión.", "Mejorar visibilidad"
        except Exception as e:
            return f"⚠️ Error en análisis visual: {str(e)}", "Indeterminado"

    score = basic_keywords_eval(user_text)
    closure_ok = detect_closure_language(user_text)
    visual_feedback, visual_eval = detect_visual_cues_from_video(video_path) if video_path else ("⚠️ Sin video disponible.", "No evaluado")

    # GPT evaluation (public)
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un coach experto en entrenamientos clínicos. Sé claro, motivador y profesional."},
                {"role": "user", "content": f"""Actúa como evaluador de una simulación médica.
Participante: {user_text}
Médico (Leo): {leo_text}
Evalúa al participante de forma motivadora y constructiva."""}
            ],
            temperature=0.4,
        )
        gpt_feedback = completion.choices[0].message.content.strip()
    except OpenAIError as e:
        gpt_feedback = f"⚠️ Evaluación GPT-4 no disponible: {str(e)}"

    public_summary = textwrap.dedent(f"""
        👏 {gpt_feedback}

        {visual_feedback}

        Áreas sugeridas:
        - Asegúrate de responder con evidencia médica.
        - Refuerza el uso del modelo de ventas Da Vinci.
        - Recuerda manejar bien cada objeción médica.
        - Mantén contacto visual con la cámara y buena presencia.
    """)

    internal_summary = textwrap.dedent(f"""
        Evaluación técnica (RH):
        - Claridad en la comunicación: {'Alta' if score >= 6 else 'Media' if score >= 3 else 'Baja'}
        - Aplicación del modelo de ventas Da Vinci: {'Mencionado' if 'da vinci' in user_text.lower() else 'No evidenciado'}
        - Dominio técnico del producto: {score}/8
        - Manejo de objeciones: {'Adecuado' if 'objeción' in user_text.lower() else 'No observado'}
        - Cierre: {'Presente' if closure_ok else 'Ausente'}
        - Evaluación visual/postural: {visual_eval}

        Recomendaciones:
        - Fortalecer argumentos clínicos y científicos.
        - Enfatizar el cierre con lenguaje de compromiso.
        - Validar objeciones planteadas por el médico antes de responder.
        - Mostrar actitud profesional ante cámara.
    """)

    return {
        "public": public_summary.strip(),
        "internal": internal_summary.strip(),
    }
