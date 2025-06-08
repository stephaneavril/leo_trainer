# evaluator.py

import os
import re
import textwrap
import cv2
import numpy as np
import json
from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_interaction(user_text, leo_text, video_path=None):
    """
    Evalúa la conversación y el lenguaje corporal del participante si hay video.
    Retorna evaluación pública (usuario) y evaluación RH (técnica).
    Ahora, se enfoca solo en el desempeño del usuario, con el diálogo de Leo como contexto mínimo si está vacío.
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
            if not cap.isOpened():
                return "⚠️ No se pudo abrir el archivo de video para análisis visual.", "Error en video", "N/A"

            frontal_frames = 0
            total_frames_processed = 0
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            max_frames_to_check = 200

            for _ in range(max_frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames_processed += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    frontal_frames += 1
            cap.release()

            if total_frames_processed == 0:
                return "⚠️ No se encontraron frames para analizar en el video.", "Sin frames", "0.0%"

            ratio = frontal_frames / total_frames_processed

            if ratio >= 0.7:
                return "✅ Te mostraste visible y profesional frente a cámara.", "Correcta", f"{ratio*100:.1f}%"
            elif ratio > 0:
                return "⚠️ Asegúrate de mantenerte visible durante toda la sesión.", "Mejorar visibilidad", f"{ratio*100:.1f}%"
            else:
                return "❌ No se detectó rostro en el video.", "No detectado", "0.0%"
        except Exception as e:
            return f"⚠️ Error en análisis visual: {str(e)}", "Error", "N/A"

    def detect_disqualifying_phrases(text):
        disqualifying_phrases = [
            "no sé", "no tengo idea", "lo invento", "no lo estudié", "no estudié bien", "no conozco", "no me acuerdo"
        ]
        return any(p in text.lower() for p in disqualifying_phrases)

    score = basic_keywords_eval(user_text)
    closure_ok = detect_closure_language(user_text)
    visual_feedback_public, visual_eval_internal, visual_pct = detect_visual_cues_from_video(video_path) if video_path and os.path.exists(video_path) else ("⚠️ Sin video disponible.", "No evaluado", "N/A")

    sales_model_score = {
        "diagnostico": any(kw in user_text.lower() for kw in ["cómo", "qué", "cuándo", "desde cuándo", "por qué", "necesita", "perfil"]),
        "argumentacion": any(kw in user_text.lower() for kw in ["beneficio", "eficaz", "estudio", "seguridad", "mecanismo", "solución", "paciente", "evidencia"]),
        "validacion": any(kw in user_text.lower() for kw in ["entiendo", "veo que", "comprendo", "es lógico", "correcto", "confirmo", "entonces"]),
        "cierre": closure_ok
    }
    model_applied_steps_count = sum(sales_model_score.values())

    active_listening_keywords = ["entiendo", "comprendo", "veo que", "lo que dices", "tiene sentido", "si entiendo bien", "parafraseando"]
    active_listening_score = sum(1 for phrase in active_listening_keywords if phrase in user_text.lower())

    feedback_level = "alto"
    gpt_public_summary = ""
    gpt_internal_structured_feedback = {}

    try:
        # TU PROMPT COMPLETO AQUÍ
        system_prompt = textwrap.dedent(""" 
Eres un evaluador experto de simulaciones de visita médica entre representantes farmacéuticos y médicos.

Tu tarea es analizar transcripciones de estas simulaciones, y generar una evaluación profesional como la que realizan coaches humanos.

Evalúa al representante según los siguientes criterios:

1️⃣ **Modelo de ventas Da Vinci**

- Diagnóstico
- Argumentación
- Validación
- Cierre

Marca una fase como "Cumplida" SOLO si el participante demuestra:

✅ conocimiento del producto  
✅ comprensión de las necesidades del médico / paciente  
✅ argumentación sólida y clínica  
✅ respuestas claras y verídicas  
✅ manejo adecuado de las preguntas del médico

Si la frase fue vaga, falsa, improvisada o sin relación con las necesidades del médico → marca "Necesita Mejora".

**Si el participante menciona información falsa, inventada o incorrecta clínicamente → marca TODAS las fases como "Necesita Mejora"**.

Ejemplos de información falsa: decir que el producto cura el cáncer, inventar estudios inexistentes, decir que reduce tumores, etc.

2️⃣ **Prioridad en uso del tiempo**

Evalúa si el participante usó bien el tiempo disponible:

- dio suficiente espacio a las necesidades del médico  
- no se extendió innecesariamente en contexto  
- priorizó la argumentación y los cierres

Marcar como: Correcta / Mejorable / Deficiente

3️⃣ **Adaptación al estilo del médico**

Evalúa si el participante:

- escuchó activamente al médico  
- adaptó su lenguaje y profundidad según el estilo del médico (más técnico, más humano, más empático, más ocupado)

Marcar como: Correcta / Mejorable / Deficiente

4️⃣ **Control de la conversación**

Evalúa si el participante:

- supo conducir la conversación  
- manejó adecuadamente tiempos y transiciones  
- realizó un cierre efectivo o no

Marcar como: Correcto / Mejorable / Deficiente

5️⃣ **Manejo de preguntas críticas del médico**

Si el médico hizo preguntas difíciles (por ejemplo: "¿por qué cambiaría a este producto?"):

- evalúa si el participante respondió de forma adecuada y convincente

Marcar como: Correcto / Mejorable / Deficiente / No aplicable (si no hubo preguntas críticas)

---

**IMPORTANTE:**

Si detectas que el participante improvisa sin conocimiento (por ejemplo: "no sé", "no lo estudié", "no tengo idea"), penaliza todas las fases.

Si detectas que el participante repite información sin estructura o de forma confusa, refleja eso en la evaluación.

Si el participante fue claro, ordenado, convincente, marca las fases como "Cumplida" con su justificación.

---

### Output esperado:

Devuelve el análisis con el siguiente formato:

```plaintext
Modelo de ventas Da Vinci:
- Diagnóstico: Cumplida / Necesita Mejora + Justificación
- Argumentación: Cumplida / Necesita Mejora + Justificación
- Validación: Cumplida / Necesita Mejora + Justificación
- Cierre: Cumplida / Necesita Mejora + Justificación

Prioridad en uso del tiempo: Correcta / Mejorable / Deficiente + Justificación

Adaptación al estilo del médico: Correcta / Mejorable / Deficiente + Justificación

Control de la conversación: Correcto / Mejorable / Deficiente + Justificación

Manejo de preguntas críticas del médico: Correcto / Mejorable / Deficiente / No aplicable + Justificación

Resumen general para RH:
[Resumen bien redactado como el que haría un coach humano.]

Áreas de mejora específicas para coaching:
[Listado de 3 a 5 recomendaciones claras, priorizadas.]
                                        
                                        Además de mostrar el análisis en formato de texto como se describió arriba, devuelve TAMBIÉN en tu respuesta final un bloque JSON llamado "internal_analysis", con la siguiente estructura y campos obligatorios:

"internal_analysis": {
    "overall_evaluation": "[Resumen general redactado como un coach humano.]",
    "Preparacion_de_la_visita": "[Cumplida / Necesita Mejora + Justificación]",
    "Apertura": "[Cumplida / Necesita Mejora + Justificación]",
    "Persuasion": "[Cumplida / Necesita Mejora + Justificación]",
    "Cierre": "[Cumplida / Necesita Mejora + Justificación]",
    "Puente": "[Cumplida / Necesita Mejora / No Aplicable + Justificación]",
    "Adaptación_al_estilo_del_medico": {
        "tipo_inferido": "[Determinado / No Determinado]",
        "nivel_de_adaptacion": "[Correcta / Mejorable / Deficiente / Necesita Mejora]",
        "comentarios": "[Comentario redactado como coach humano]"
    },
    "Manejo_de_objeciones": "[Correcto / Mejorable / Deficiente / No Aplicable + Justificación]",
    "Escucha_activa": "[Cumplida / Necesita Mejora + Justificación]",
    "Areas_de_mejora": [
        "Primera recomendación clara",
        "Segunda recomendación clara",
        "Tercera recomendación clara",
        "Cuarta recomendación clara",
        "Quinta recomendación clara"
    ]
}
                                        
 """)  

        if not leo_text.strip():
            user_prompt = f"""
                --- Inicio de Simulación ---
                Participante (Tú): {user_text}
                Médico (Leo): (Diálogo del agente no disponible para análisis)
                --- Fin de Simulación ---

                Por favor, proporciona una evaluación detallada del Participante en formato JSON, siguiendo las instrucciones de tu rol como coach experto en ventas farmacéuticas. Enfócate exclusivamente en las palabras del Participante.
                Además, incluye en el JSON final el bloque "internal_analysis" con la estructura exacta que se indicó.
                """
        else:
            user_prompt = f"""
                --- Inicio de Simulación ---
                Participante (Tú): {user_text}
                Médico (Leo): {leo_text}
                --- Fin de Simulación ---

                Por favor, proporciona una evaluación detallada del Participante en formato JSON, siguiendo las instrucciones de tu rol como coach experto en ventas farmacéuticas.
                """

        # Detectar frases descalificadoras antes de llamar a GPT
        disqualifying_flag = detect_disqualifying_phrases(user_text)

        if disqualifying_flag:
            print("[INFO] Se detectaron frases descalificadoras en el texto del participante.")

        # Llamar a GPT
        completion = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
        )

        gpt_response_content = completion.choices[0].message.content.strip()

        try:
            parsed_gpt_response = json.loads(gpt_response_content)
            gpt_public_summary = parsed_gpt_response.get("public_summary", "No se generó resumen público.")
            gpt_internal_structured_feedback = parsed_gpt_response.get("internal_analysis", {})
        except json.JSONDecodeError:
            print(f"[ERROR] GPT response was not valid JSON: {gpt_response_content}")
            gpt_public_summary = "⚠️ Evaluación automática (GPT) no disponible: Formato inválido."
            gpt_internal_structured_feedback = {"error": "Formato JSON inválido de GPT.", "raw_response": gpt_response_content[:200]}

    except OpenAIError as e:
        gpt_public_summary = f"⚠️ Evaluación automática (GPT) no disponible en este momento debido a un error: {str(e)}"
        gpt_internal_structured_feedback = {"error": f"Error de OpenAI: {str(e)}"}
        feedback_level = "error"
    except Exception as e:
        gpt_public_summary = f"⚠️ Evaluación automática (GPT) no disponible debido a un error inesperado: {str(e)}"
        gpt_internal_structured_feedback = {"error": f"Error inesperado al llamar a GPT: {str(e)}"}
        feedback_level = "error"

    final_internal_summary_dict = {
        "overall_rh_summary": gpt_internal_structured_feedback.get("overall_evaluation", "Evaluación general no disponible del GPT."),
        "knowledge_score": f"{score}/8",
        "visual_presence": visual_eval_internal,
        "visual_percentage": visual_pct,
        "sales_model_simple_detection": {
            "diagnostico": '✅' if sales_model_score['diagnostico'] else '❌',
            "argumentacion": '✅' if sales_model_score['argumentacion'] else '❌',
            "validacion": '✅' if sales_model_score['validacion'] else '❌',
            "cierre": '✅' if sales_model_score['cierre'] else '❌',
            "steps_applied_count": f"{model_applied_steps_count}/4"
        },
        "active_listening_simple_detection": 'Alta' if active_listening_score >= 4 else 'Moderada' if active_listening_score >= 2 else 'Baja',
        "disqualifying_phrases_detected": disqualifying_flag,
        "gpt_detailed_feedback": gpt_internal_structured_feedback,
        "error_during_eval": gpt_internal_structured_feedback.get("error", "No error detected from GPT.")
    }

    public_summary_for_user = textwrap.dedent(f"""
        {gpt_public_summary}

        {visual_feedback_public}

        Áreas sugeridas adicionales:
        - Asegúrate de responder con evidencia médica sólida y adaptada a la necesidad del médico.
        - Refuerza el uso estructurado del modelo de ventas Da Vinci en cada interacción.
        - Practica el manejo de objeciones aplicando la metodología APACT.
        - Mantén un contacto visual adecuado con la cámara y una buena presencia general durante la sesión.
    """)

    return {
        "public": public_summary_for_user.strip(),
        "internal": final_internal_summary_dict,
        "level": feedback_level
    }
