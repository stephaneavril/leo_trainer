# evaluator.py

import os
import openai
import textwrap
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_interaction(user_text, leo_text):
    """
    Evalúa la interacción con base en los textos proporcionados.
    Si está disponible la API de OpenAI, utiliza GPT-4 para redacción fluida.
    Además incluye análisis técnicos robustos locales para RH.
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

    def detect_visual_cues(text):
        if re.search(r"(c[aá]mara|postura|frente al m[oó]dulo|me ves|visible|pantalla)", text, re.IGNORECASE):
            return "✅ Tu postura fue profesional y te mostraste frente a la cámara con claridad.", "Correcta"
        return "⚠️ Asegúrate de mantener una postura profesional y estar visible correctamente frente a la cámara.", "Mejorar visibilidad frente a cámara"

    # Análisis técnico local
    score = basic_keywords_eval(user_text)
    closure_ok = detect_closure_language(user_text)
    visual_feedback, visual_eval = detect_visual_cues(user_text)

    # Redacción motivacional GPT-4 si es posible
    gpt_feedback = ""
    try:
        if openai.api_key and openai.api_key.startswith("sk-"):
            prompt = f"""
            Actúa como evaluador de una simulación médica.
            Participante: {user_text}
            Médico (Leo): {leo_text}
            Evalúa al participante de forma constructiva, motivadora y profesional.
            """
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un coach experto en entrenamientos clínicos. Sé específico y claro."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.5
            )
            gpt_feedback = res.choices[0].message.content.strip()
        else:
            gpt_feedback = "Gracias por tu participación. Mostraste buena disposición. Sigue mejorando tus habilidades clínicas y de comunicación."
    except Exception as e:
        gpt_feedback = f"⚠️ Evaluación GPT-4 no disponible: {str(e)}"

    public_summary = textwrap.dedent(f"""
        👏 {gpt_feedback}

        {visual_feedback}

        Áreas sugeridas:
        - Asegúrate de responder con evidencia médica.
        - Refuerza el uso del modelo de ventas Da Vinci.
        - Recuerda manejar bien cada objeción médica.
        - Mantén contacto visual con la cámara y buena presencia.
    ")

    internal_summary = textwrap.dedent(f"""
            Evaluación técnica (RH):
        - Claridad en la comunicación: {'Alta' if score >= 6 else 'Media' if score >= 3 else 'Baja'}
        - Aplicación del modelo de ventas Da Vinci: {'Mencionado' if 'da vinci' in user_text.lower() else 'No evidenciado'}
        - Dominio técnico del producto: {score}/8
        - Manejo de objeciones: {'Adecuado' if 'objeción' in user_text.lower() else 'No observado'}
        - Cierre: {'Presente' if closure_ok else 'Ausente'}
        - Evaluación visual/postural: {visual_eval}

        Recomendaciones:
        - Fortalecer argumentos clínicos y científicos con datos de estudios o mecanismos.
        - Enfatizar el cierre con lenguaje de compromiso.
        - Validar objeciones planteadas por el médico antes de responder.
        - Mostrar actitud profesional ante cámara.
    ")

    return {
        "public": public_summary.strip(),
        "internal": internal_summary.strip(),
    }
