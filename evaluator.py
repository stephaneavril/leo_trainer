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

def evaluate_interaction(user_text, leo_text, video_path=None): # leo_text will be empty now
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

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frontal_frames = 0
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            frames_to_analyze = min(200, frame_count) # Limit frames to analyze for performance

            if frames_to_analyze == 0:
                cap.release()
                return "⚠️ No se encontraron frames para analizar en el video.", "Sin frames", "0.0%"

            for _ in range(frames_to_analyze):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    frontal_frames += 1
            cap.release()
            
            if frontal_frames > 0:
                ratio = frontal_frames / frames_to_analyze
            else:
                ratio = 0

            if ratio >= 0.7:
                return "✅ Te mostraste visible y profesional frente a cámara.", "Correcta", f"{ratio*100:.1f}%"
            elif ratio > 0:
                return "⚠️ Asegúrate de mantenerte visible durante toda la sesión.", "Mejorar visibilidad", f"{ratio*100:.1f}%"
            else:
                return "❌ No se detectó rostro en el video.", "No detectado", "0.0%"
        except Exception as e:
            return f"⚠️ Error en análisis visual: {str(e)}", "Error", "N/A"

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


    # GPT feedback
    feedback_level = "alto"
    gpt_public_summary = ""
    gpt_internal_structured_feedback = {}

    try:
        # Request a JSON output from GPT
        # Adjusting the system prompt to focus on user behavior as Leo's dialogue is not provided.
        # Removing any specific mentions of Leo's text for evaluation context.
        system_prompt = textwrap.dedent("""
            Eres un coach experto en entrenamiento de ventas farmacéuticas para representantes médicos.
            Tu objetivo es evaluar el desempeño del "Participante" en una simulación de visita médica,
            aplicando el 'Modelo de Ventas Da Vinci' y los principios de 'Insights Discovery' (comunicación adaptativa).
            Proporciona retroalimentación constructiva, específica y accionable.

            La evaluación debe centrarse exclusivamente en la actuación del Participante. El "Médico (Leo)" es un avatar interactivo cuyo diálogo no se proporciona para el análisis.

            **Modelo de Ventas Da Vinci - Fases Clave:**
            1.  **Preparación de la Visita (Paso 1)**: Evalúa si el participante demuestra haber analizado la información y fijado objetivos (evidenciado en su discurso inicial o preguntas).
            2.  **Apertura (Paso 2)**: ¿El participante capturó la atención del médico y despertó su interés? ¿Creó un vínculo? (Ej: saludo profesional, mención de visita anterior, enfoque en el paciente/síntomas relevantes).
            3.  **Persuasión (Paso 3)**: ¿Descubrió necesidades del paciente y transmitió mensajes clave para persuadir al médico? ¿Utilizó preguntas poderosas? ¿Presentó beneficios que conectan con las necesidades del médico/paciente? ¿Manejó objeciones?
            4.  **Cierre (Paso 4)**: ¿Realizó acuerdos de prescripción? ¿Identificó señales de compra y solicitó el uso del producto?
            5.  **Puente (Paso 5 - para segundo producto, si aplica)**: Si la conversación da pie a un segundo producto, ¿el participante realizó una transición efectiva usando "ganchos"?
            6.  **Análisis Posterior (Paso 8)**: Aunque no se evalúa directamente en la conversación, infiere si el participante demostró capacidad para auto-evaluar la visita.

            **Habilidades/Competencias Clave (transversales al modelo):**
            -   **Diagnóstico Profundo (D)**: Capacidad de indagar en las necesidades del médico y perfil del paciente.
            -   **Arte de Conectar (A)**: Empatía, escucha activa, creación de conversaciones significativas.
            -   **Valor con Propósito (V)**: Presentar soluciones relevantes, centradas en el bienestar del paciente y con evidencia.
            -   **Innovación Adaptativa (I)**: Personalización del approach, uso creativo de herramientas (aunque no observables aquí).
            -   **Nutrir el Conocimiento (N)**: Demostrar conocimiento profundo del producto y patología.
            -   **Curiosidad Activa (C)**: Preguntar, investigar, cuestionar para ir más allá de lo evidente.
            -   **Impacto Positivo (I)**: Dejar una huella memorable y un beneficio real.

            **Insights Discovery - Adaptación al estilo del Médico (Leo):**
            -   Evalúa si el participante adaptó su comunicación basándose en su discurso. Si es posible, infiere el tipo de personalidad del médico (Rojo Fuego, Azul Mar, Amarillo Sol, Verde Tierra) basado solo en las preguntas o inferencias del participante sobre el médico y evalúa la adaptación.

            **Manejo de Objeciones (MILD/APACT):**
            -   Si el participante mencionó una objeción del médico, ¿el participante la manejó siguiendo los pasos de APACT (Admitir, Preguntar, Argumentar, Confirmar, Transición) y si identificó el tipo de objeción (Malentendido, Indiferencia, Limitación, Duda - MILD)? Si no se mencionó ninguna objeción del médico por parte del participante, asume que no hubo.

            **Output Format:**
            Your response MUST be a JSON object with two main keys: "public_summary" (string) and "internal_analysis" (object).

            The "public_summary" should be a concise, motivating overall feedback for the participant.
            The "internal_analysis" object should contain detailed, structured feedback for HR, with keys like:
            - "overall_evaluation": string (brief summary for HR)
            - "da_vinci_model_feedback": object with keys for each phase (e.g., "apertura": {"score": "Bueno/Regular/Necesita Mejora", "feedback": "Detalle específico"})
            - "insights_discovery_adaptation": {"inferred_leo_type": "string", "adaptation_score": "Bueno/Regular/Necesita Mejora", "feedback": "Detalle específico sobre la adaptación"}
            - "objection_handling_feedback": {"objection_detected": "bool", "type": "string", "apact_applied": "string (Total/Parcial/No Aplicado)", "feedback": "Detalle específico"}
            - "active_listening_feedback": {"score": "Bueno/Regular/Necesita Mejora", "feedback": "Detalle específico"}
            - "strengths": array of strings
            - "areas_for_improvement_specific": array of strings (actionable advice for HR)
        """)

        # If leo_text is empty, adjust the user prompt to reflect that context
        if not leo_text.strip():
            user_prompt = f"""
                --- Inicio de Simulación ---
                Participante (Tú): {user_text}
                Médico (Leo): (Diálogo del agente no disponible para análisis)
                --- Fin de Simulación ---

                Por favor, proporciona una evaluación detallada del Participante en formato JSON, siguiendo las instrucciones de tu rol como coach experto en ventas farmacéuticas. Enfócate exclusivamente en las palabras del Participante.
                """
        else:
            user_prompt = f"""
                --- Inicio de Simulación ---
                Participante (Tú): {user_text}
                Médico (Leo): {leo_text}
                --- Fin de Simulación ---

                Por favor, proporciona una evaluación detallada del Participante en formato JSON, siguiendo las instrucciones de tu rol como coach experto en ventas farmacéuticas.
                """

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

        "gpt_detailed_feedback": gpt_internal_structured_feedback,
        "error_during_eval": gpt_internal_structured_feedback.get("error", "No error detected from GPT.")
    }
    
    final_internal_summary_json = json.dumps(final_internal_summary_dict, ensure_ascii=False, indent=2)


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
        "internal": final_internal_summary_json,
        "level": feedback_level
    }