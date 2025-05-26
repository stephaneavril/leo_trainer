import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_interaction(message, response):
    """
    Evalúa automáticamente el desempeño usando OpenAI si es posible,
    y con lógica local si no.
    """
    full_text = f"Usuario: {message}\nLeo: {response}"

    if not openai.api_key or openai.api_key.startswith("sk-") is False:
        return local_evaluation(message)

    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un evaluador de desempeño en simulaciones de ventas. Evalúa las respuestas del participante y ofrece retroalimentación constructiva."},
                {"role": "user", "content": full_text}
            ],
            temperature=0.5
        )

        content = res.choices[0].message.content.strip()

        return {
            "public": "✅ " + content,
            "internal": "🔍 Evaluación IA generada por modelo GPT-4."
        }

    except Exception as e:
        return {
            "public": "⚠️ Evaluación automática no disponible.",
            "internal": f"❌ Error de evaluación con OpenAI: {str(e)}"
        }

def local_evaluation(message):
    criterios = {
        "claridad": "Demostró claridad en sus respuestas",
        "objeciones": "Manejó adecuadamente las objeciones",
        "modelo_ventas": "Aplicó el modelo de ventas correctamente",
        "cierre": "Logró un cierre o compromiso"
    }

    evaluacion = []

    if any(word in message.lower() for word in ["claro", "entiendo", "explicar"]):
        evaluacion.append(criterios["claridad"])
    else:
        evaluacion.append("Debe mejorar la claridad en sus respuestas")

    if any(word in message.lower() for word in ["objeción", "duda", "preocupación"]):
        evaluacion.append(criterios["objeciones"])
    else:
        evaluacion.append("No se identificó manejo de objeciones")

    if "modelo" in message.lower():
        evaluacion.append(criterios["modelo_ventas"])
    else:
        evaluacion.append("No aplicó claramente el modelo de ventas")

    if any(word in message.lower() for word in ["compromiso", "prescribir", "recomendar"]):
        evaluacion.append(criterios["cierre"])
    else:
        evaluacion.append("Faltó un cierre o compromiso")

    return {
        "public": " • ".join(evaluacion),
        "internal": "Evaluación generada localmente (palabras clave)."
    }
