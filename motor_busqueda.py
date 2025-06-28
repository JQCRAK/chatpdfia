"""
Módulo para realizar búsqueda semántica y generar respuestas
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from generador_embeddings import generate_single_embedding
import re


def detect_intent(question):
    """
    Detecta la intención del usuario para responder apropiadamente
    
    Args:
        question (str): Pregunta del usuario
        
    Returns:
        dict: Información sobre la intención detectada
    """
    question_lower = question.lower().strip()
    
    # Patrones para diferentes intenciones
    patterns = {
        'greeting': [
            r'\b(hola|buenos días|buenas tardes|buenas noches|hey|hi|hello)\b',
            r'\b(¿?cómo estás?|¿?como estas?|¿?qué tal?|¿?que tal?)\b',
            r'^(hi|hello|hola)$'
        ],
        'identity': [
            r'\b(¿?cuál es tu nombre|¿?cual es tu nombre|¿?cómo te llamas|¿?como te llamas)\b',
            r'\b(¿?quién eres|¿?quien eres|¿?qué eres|¿?que eres)\b',
            r'\b(tu nombre|your name|dime tu nombre)\b'
        ],
        'capabilities': [
            r'\b(¿?qué puedes hacer|¿?que puedes hacer|¿?qué sabes hacer|¿?que sabes hacer)\b',
            r'\b(¿?en qué me puedes ayudar|¿?en que me puedes ayudar)\b',
            r'\b(ayúdame|ayudame|help me|ayuda)\b',
            r'\b(¿?para qué sirves|¿?para que sirves)\b'
        ],
        'creators': [
            r'\b(¿?quién te creó|¿?quien te creo|¿?quiénes son tus creadores|¿?quienes son tus creadores)\b',
            r'\b(¿?quién te hizo|¿?quien te hizo|¿?quién te programó|¿?quien te programo)\b',
            r'\b(creadores|desarrolladores|programadores)\b',
            r'\b(¿?quién está detrás de ti|¿?quien esta detras de ti)\b'
        ],
        'thanks': [
            r'\b(gracias|thank you|thanks|muchas gracias)\b',
            r'\b(te agradezco|agradecido)\b'
        ],
        'goodbye': [
            r'\b(adiós|adios|bye|hasta luego|nos vemos|chao|goodbye)\b',
            r'\b(hasta la vista|hasta pronto)\b'
        ]
    }
    
    # Verificar cada patrón
    for intent, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, question_lower):
                return {
                    'intent': intent,
                    'confidence': 0.9,
                    'requires_document': False
                }
    
    # Si no es una intención específica, es una pregunta sobre documento
    return {
        'intent': 'document_query',
        'confidence': 0.8,
        'requires_document': True
    }


def generate_intent_response(intent_info, question):
    """
    Genera respuestas para intenciones específicas (no relacionadas con documentos)
    
    Args:
        intent_info (dict): Información sobre la intención
        question (str): Pregunta original
        
    Returns:
        str: Respuesta generada
    """
    intent = intent_info['intent']
    
    responses = {
        'greeting': [
            "¡Hola! 👋 Soy Jahr, tu asistente inteligente para documentos PDF. ¿Cómo estás?",
            "¡Hola! 😊 Me alegra saludarte. Estoy aquí para ayudarte con tus documentos PDF.",
            "¡Buenos días! ☀️ Soy Jahr, ¿en qué puedo ayudarte hoy?"
        ],
        
        'identity': [
            "Soy **Jahr**, tu asistente inteligente especializado en analizar documentos PDF. 🤖\n\n" +
            "Mi función es ayudarte a encontrar información específica en tus documentos y responder preguntas sobre su contenido.",
            
            "Mi nombre es **Jahr** 🤖 y soy un chatbot diseñado para trabajar con documentos PDF.\n\n" +
            "Puedo leer, analizar y responder preguntas sobre cualquier PDF que subas."
        ],
        
        'capabilities': [
            "¡Excelente pregunta! Estas son mis principales capacidades: 🚀\n\n" +
            "📄 **Análisis de PDFs**: Leo y proceso documentos PDF\n" +
            "🔍 **Búsqueda inteligente**: Encuentro información específica\n" +
            "💬 **Respuestas precisas**: Genero respuestas basadas en el documento\n" +
            "📝 **Resúmenes**: Puedo resumir secciones o todo el documento\n" +
            "❓ **Preguntas específicas**: Respondo sobre temas particulares del PDF\n\n" +
            "¡Sube un PDF y podrás preguntarme lo que necesites sobre él!",
            
            "Puedo ayudarte de muchas formas: 💪\n\n" +
            "• Analizar documentos PDF completos\n" +
            "• Responder preguntas específicas sobre el contenido\n" +
            "• Crear resúmenes de los puntos principales\n" +
            "• Encontrar información particular que necesites\n" +
            "• Explicar conceptos complejos del documento\n\n" +
            "¿Tienes algún PDF que quieras que analice?"
        ],
        
        'creators': [
            "¡Me enorgullece presentar a mi increíble equipo de creadores! 👨‍💻👩‍💻\n\n" +
            "🏆 **Mis creadores son:**\n" +
            "• **Jhostin Quispe**\n" +
            "• **Angel Guaño**\n" +
            "• **Rumi Grefa**\n" +
            "• **Henry Reding**\n\n" +
            "Este talentoso equipo me diseñó y programó para ser tu mejor asistente PDF. ¡Son geniales! 🌟",
            
            "Fui creado por un excelente equipo de desarrolladores: 💻✨\n\n" +
            "👥 **Equipo Jahr:**\n" +
            "- Jhostin Quispe\n" +
            "- Angel Guaño  \n" +
            "- Rumi Grefa\n" +
            "- Henry Reding\n\n" +
            "Gracias a ellos puedo ayudarte con tus documentos PDF de manera inteligente."
        ],
        
        'thanks': [
            "¡De nada! 😊 Estoy aquí para ayudarte siempre que lo necesites.",
            "¡Un placer ayudarte! 🤗 ¿Hay algo más en lo que pueda asistirte?",
            "¡Me alegra haberte sido útil! 😄 No dudes en preguntarme lo que necesites."
        ],
        
        'goodbye': [
            "¡Hasta luego! 👋 Fue un placer ayudarte. Vuelve cuando necesites analizar más documentos.",
            "¡Adiós! 😊 Que tengas un excelente día. Aquí estaré cuando regreses.",
            "¡Nos vemos pronto! 🌟 Gracias por usar Jahr."
        ]
    }
    
    import random
    return random.choice(responses.get(intent, ["¡Hola! ¿En qué puedo ayudarte?"]))


def search_similar_chunks(question, chunks, embeddings, threshold=0.2, top_k=3):
    """
    Busca los fragmentos más similares a una pregunta usando similitud coseno
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        embeddings (np.array): Array de embeddings de los fragmentos
        threshold (float): Umbral mínimo de similitud
        top_k (int): Número máximo de fragmentos a retornar
        
    Returns:
        tuple: (fragmentos_relevantes, puntuaciones)
    """
    if len(chunks) == 0 or embeddings is None or len(embeddings) == 0:
        return [], []
    
    # Generar embedding de la pregunta
    question_embedding = generate_single_embedding(question)
    if question_embedding is None:
        return [], []
    
    # Calcular similitudes
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    
    # Obtener índices ordenados por similitud
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    relevant_chunks = []
    relevant_scores = []
    
    for idx in top_indices:
        if similarities[idx] >= threshold:
            relevant_chunks.append(chunks[idx])
            relevant_scores.append(similarities[idx])
    
    return relevant_chunks, relevant_scores


def generate_response(question, relevant_chunks, scores, intent_info=None):
    """
    Genera una respuesta basada en los fragmentos relevantes encontrados o la intención detectada
    
    Args:
        question (str): Pregunta del usuario
        relevant_chunks (list): Fragmentos relevantes encontrados
        scores (list): Puntuaciones de similitud
        intent_info (dict): Información sobre la intención detectada
        
    Returns:
        str: Respuesta generada
    """
    # Si hay información de intención y no requiere documento
    if intent_info and not intent_info.get('requires_document', True):
        return generate_intent_response(intent_info, question)
    
    # Si no hay chunks relevantes
    if not relevant_chunks:
        return "🔍 No encontré información específica sobre esa pregunta en el documento. Intenta reformular tu pregunta o pregunta sobre otros temas del PDF."
    
    # Respuesta con alta confianza (> 0.5)
    if scores[0] > 0.5:
        if len(relevant_chunks) == 1:
            return f"📖 **Según el documento:**\n\n{relevant_chunks[0]}"
        else:
            response = "📖 **Información relevante encontrada:**\n\n"
            for i, chunk in enumerate(relevant_chunks[:2]):  # Máximo 2 chunks
                response += f"**{i+1}.** {chunk}\n\n"
            return response
    
    # Respuesta con confianza media (0.3-0.5)
    elif scores[0] > 0.3:
        return f"📖 **Posible información relacionada:**\n\n{relevant_chunks[0]}\n\n*Nota: Esta información podría estar relacionada con tu pregunta.*"
    
    # Respuesta con baja confianza (0.2-0.3)
    else:
        return f"🔍 **Información parcialmente relacionada:**\n\n{relevant_chunks[0]}\n\n*Nota: No estoy completamente seguro de que esto responda tu pregunta específica.*"


def get_search_statistics(scores):
    """
    Obtiene estadísticas de la búsqueda realizada
    
    Args:
        scores (list): Lista de puntuaciones de similitud
        
    Returns:
        dict: Diccionario con estadísticas
    """
    if not scores:
        return {
            'max_score': 0,
            'avg_score': 0,
            'total_matches': 0,
            'confidence_level': 'none'
        }
    
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    # Determinar nivel de confianza
    if max_score > 0.5:
        confidence_level = 'high'
    elif max_score > 0.3:
        confidence_level = 'medium'
    elif max_score > 0.2:
        confidence_level = 'low'
    else:
        confidence_level = 'very_low'
    
    return {
        'max_score': max_score,
        'avg_score': avg_score,
        'total_matches': len(scores),
        'confidence_level': confidence_level
    }


def improve_question(question):
    """
    Mejora la pregunta del usuario para obtener mejores resultados
    
    Args:
        question (str): Pregunta original
        
    Returns:
        str: Pregunta mejorada
    """
    # Normalizar la pregunta
    question = question.strip().lower()
    
    # Expandir abreviaciones comunes
    replacements = {
        'q ': 'que ',
        'xq': 'por que',
        'x': 'por',
        'k ': 'que '
    }
    
    for old, new in replacements.items():
        question = question.replace(old, new)
    
    return question


def process_user_query(question, chunks=None, embeddings=None):
    """
    Procesa una consulta del usuario de manera integral
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        embeddings (np.array): Array de embeddings
        
    Returns:
        str: Respuesta final
    """
    # Detectar intención
    intent_info = detect_intent(question)
    
    # Si no requiere documento, generar respuesta directa
    if not intent_info.get('requires_document', True):
        return generate_intent_response(intent_info, question)
    
    # Si requiere documento pero no hay chunks
    if not chunks or embeddings is None or len(embeddings) == 0:
        return "📁 Para responder preguntas sobre documentos, primero necesito que subas un archivo PDF."
    
    # Mejorar la pregunta
    improved_question = improve_question(question)
    
    # Buscar fragmentos relevantes
    relevant_chunks, scores = search_similar_chunks(
        improved_question, chunks, embeddings
    )
    
    # Generar respuesta
    return generate_response(question, relevant_chunks, scores, intent_info)