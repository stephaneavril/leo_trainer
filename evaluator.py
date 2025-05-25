def evaluate_interaction(message, response):
    """
    Evalúa automáticamente el desempeño del participante en base a su mensaje y la respuesta de Leo.
    Retorna una evaluación breve para mostrar en el panel de RH.
    """
    criterios = {
        "claridad": "Demostró claridad en sus respuestas",
        "objeciones": "Manejó adecuadamente las objeciones",
        "modelo_ventas": "Aplicó el modelo de ventas correctamente",
        "cierre": "Logró un cierre o compromiso"
    }

    # Simulación simple de evaluación basada en palabras clave (se puede reemplazar por LLM real)
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

    return " • ".join(evaluacion)
