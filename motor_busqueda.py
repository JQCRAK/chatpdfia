"""
Módulo para realizar búsqueda semántica y generar respuestas
NOW USING TRUE SEMANTIC UNDERSTANDING - NO MORE TEMPLATE MATCHING
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from generador_embeddings import generate_single_embedding
from semantic_analyzer import TrueSemanticPDFAnalyzer
from enhanced_search_engine import EnhancedSearchEngine
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


def analyze_line_by_line(question, chunks):
    """
    Analiza línea por línea dentro de cada chunk para encontrar la mejor coincidencia literal y precisa
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        
    Returns:
        tuple: (fragmentos_relevantes, puntuaciones, líneas_específicas)
    """
    if not chunks or not question:
        return [], [], []

    question_lower = question.lower().strip()
    question_words = set(question_lower.split())
    stop_words = {
        'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para',
        'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the',
        'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as', 'on',
        'at', 'about', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'
    }
    meaningful_words = question_words - stop_words

    best_line = ""
    best_score = 0
    best_chunk = None
    for chunk in chunks:
        lines = [line.strip() for line in chunk.split('\n') if line.strip()]
        for line in lines:
            line_lower = line.lower()
            line_words = set(line_lower.split())
            # Score: overlap de palabras significativas, penalizar líneas largas
            overlap = len(meaningful_words.intersection(line_words))
            length_penalty = 1.0 if 8 <= len(line.split()) <= 40 else 0.5  # Prefiere frases de 8 a 40 palabras
            score = overlap * length_penalty
            # Bonus si la línea contiene todas las palabras significativas
            if overlap == len(meaningful_words) and len(meaningful_words) > 0:
                score += 2
            # Penalizar si la línea es muy larga (>40 palabras)
            if len(line.split()) > 40:
                score *= 0.3
            # Penalizar si la línea es muy corta (<8 palabras)
            if len(line.split()) < 8:
                score *= 0.2
            # Penalizar si es mayúsculas o metadatos
            if line.isupper() or 'isbn' in line_lower:
                score *= 0.1
            if score > best_score:
                best_score = score
                best_line = line
                best_chunk = chunk
    # Si se encontró una línea suficientemente relevante, devolver solo esa
    if best_line and best_score > 0.5:
        return [best_chunk], [best_score], [best_line]
    # Si no, devolver vacío para que el flujo use embeddings
    return [], [], []


def search_with_cross_validation(question, chunks, embedding_models, thresholds, top_k=3):
    """
    Búsqueda basada en múltiples modelos de embeddings y validación cruzada
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        embedding_models (list): Modelos de embeddings a usar
        thresholds (list): Lista de umbrales de similitud
        top_k (int): Número de resultados a retornar
        
    Returns:
        tuple: (chunks relevantes, puntuaciones, líneas específicas)
    """
    # Primero hacer búsqueda línea por línea para encontrar coincidencias exactas
    relevant_chunks, relevance_scores, specific_lines = analyze_line_by_line(question, chunks)
    
    # Si no hay resultados exactos, usar embeddings
    if not relevant_chunks:
        # Generar embedding de la pregunta
        question_embedding = generate_single_embedding(question)
        
        # Buscar chunks similares usando cada modelo
        all_results = []
        for model, threshold in zip(embedding_models, thresholds):
            try:
                # Generar embeddings de chunks con el modelo actual
                chunk_embeddings = model.encode(chunks)
                
                # Calcular similitud
                similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
                
                # Filtrar por umbral
                for idx, score in enumerate(similarities):
                    if score > threshold:
                        all_results.append((chunks[idx], score, chunks[idx]))
                        
            except Exception as e:
                continue
        
        # Ordenar resultados por puntuación
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar los top_k mejores resultados
        if all_results:
            relevant_chunks = [r[0] for r in all_results[:top_k]]
            relevance_scores = [r[1] for r in all_results[:top_k]]
            specific_lines = [r[2] for r in all_results[:top_k]]
    
    return relevant_chunks, relevance_scores, specific_lines


def validate_semantic_match(question, chunk, similarity_score):
    """
    Valida que realmente haya una relación semántica entre la pregunta y el chunk
    """
    question_words = set(question.lower().split())
    chunk_words = set(chunk.lower().split())
    
    # Filtrar palabras vacías
    stop_words = {'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para', 'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as', 'al', 'lo', 'le', 'su', 'sus', 'esta', 'este', 'esto', 'esa', 'ese', 'eso'}
    
    meaningful_question_words = question_words - stop_words
    meaningful_chunk_words = chunk_words - stop_words
    
    # Si no hay palabras significativas en la pregunta, no validar
    if len(meaningful_question_words) == 0:
        return True
    
    # Calcular intersección de palabras significativas
    word_overlap = len(meaningful_question_words.intersection(meaningful_chunk_words))
    word_overlap_ratio = word_overlap / len(meaningful_question_words)
    
    # Validación estricta: debe haber tanto similitud semántica como palabras en común
    return similarity_score > 0.6 and word_overlap_ratio > 0.3


def search_exact_matches(question, chunks):
    """
    Busca coincidencias exactas para nombres, fechas, números y términos específicos
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        
    Returns:
        list: Lista de chunks que contienen coincidencias exactas
    """
    if not chunks:
        return []
    
    question_lower = question.lower().strip()
    exact_matches = []
    
    # Extraer términos específicos de la pregunta
    # Fechas (años)
    years = re.findall(r'\b(19|20)\d{2}\b', question)
    # Nombres propios (capitalizados)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    # Números específicos
    numbers = re.findall(r'\b\d+\b', question)
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Verificar fechas/años
        for year in years:
            if year in chunk_lower:
                exact_matches.append(chunk)
                break
        
        # Verificar nombres propios
        for name in names:
            if name.lower() in chunk_lower:
                exact_matches.append(chunk)
                break
        
        # Verificar números específicos
        for number in numbers:
            if number in chunk_lower:
                exact_matches.append(chunk)
                break
        
        # Verificar palabras clave específicas de la pregunta
        question_words = question_lower.split()
        significant_words = [word for word in question_words if len(word) > 4]
        
        for word in significant_words:
            if word in chunk_lower:
                # Verificar que no sea una coincidencia muy común
                if chunk not in exact_matches:
                    exact_matches.append(chunk)
                break
    
    return exact_matches


def calculate_text_match_score(question, chunk):
    """
    Calcula un score de coincidencia textual entre pregunta y chunk
    
    Args:
        question (str): Pregunta normalizada
        chunk (str): Chunk normalizado
        
    Returns:
        float: Score de coincidencia textual (0-1)
    """
    if not question or not chunk:
        return 0.0
    
    question_words = set(question.split())
    chunk_words = set(chunk.split())
    
    # Filtrar palabras comunes
    stop_words = {'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para', 'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as'}
    
    meaningful_question_words = question_words - stop_words
    meaningful_chunk_words = chunk_words - stop_words
    
    if not meaningful_question_words:
        return 0.0
    
    # Calcular overlap básico
    word_overlap = len(meaningful_question_words.intersection(meaningful_chunk_words))
    basic_score = word_overlap / len(meaningful_question_words)
    
    # Bonus por coincidencias exactas de términos largos
    exact_bonus = 0
    for word in meaningful_question_words:
        if len(word) > 4 and word in chunk:
            exact_bonus += 0.2
    
    # Bonus por fechas, números, nombres propios
    special_bonus = 0
    # Años
    years_in_question = re.findall(r'\b(19|20)\d{2}\b', question)
    for year in years_in_question:
        if year in chunk:
            special_bonus += 0.5
    
    # Nombres propios (palabras capitalizadas en el chunk original)
    names_in_question = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    for name in names_in_question:
        if name.lower() in chunk:
            special_bonus += 0.4
    
    # Score final
    final_score = min(1.0, basic_score + exact_bonus + special_bonus)
    return final_score


def validate_semantic_match_flexible(question, chunk, similarity_score):
    """
    Validación semántica balanceada: estricta para preguntas irrelevantes, flexible para contenido del documento
    """
    question_words = set(question.lower().split())
    chunk_words = set(chunk.lower().split())
    
    # Filtrar palabras vacías
    stop_words = {'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para', 'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as', 'al', 'lo', 'le', 'su', 'sus', 'esta', 'este', 'esto', 'esa', 'ese', 'eso'}
    
    meaningful_question_words = question_words - stop_words
    meaningful_chunk_words = chunk_words - stop_words
    
    if len(meaningful_question_words) == 0:
        return similarity_score > 0.5  # Permitir si hay buena similitud semántica
    
    # Calcular intersección de palabras significativas
    word_overlap = len(meaningful_question_words.intersection(meaningful_chunk_words))
    word_overlap_ratio = word_overlap / len(meaningful_question_words)
    
    # Validación balanceada:
    # - Si hay alta similitud semántica (>0.5), permitir con menos overlap
    # - Si hay buen overlap de palabras (>0.2), permitir con menor similitud
    # - Requerir al menos una palabra en común para preguntas específicas
    return ((similarity_score > 0.5 and word_overlap_ratio > 0.1) or 
            (similarity_score > 0.4 and word_overlap_ratio > 0.2) or
            (similarity_score > 0.6))


def search_similar_chunks(question, chunks, embeddings, threshold=0.3, top_k=3):
    """
    Búsqueda semántica mejorada que combina similitud semántica con búsqueda textual
    
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
    
    # Primero intentar búsqueda textual directa para nombres, fechas, etc.
    exact_matches = search_exact_matches(question, chunks)
    if exact_matches:
        return exact_matches[:top_k], [0.9] * len(exact_matches[:top_k])
    
    # Generar embedding de la pregunta
    question_embedding = generate_single_embedding(question)
    if question_embedding is None:
        return [], []
    
    # Calcular similitudes coseno
    try:
        similarities = cosine_similarity([question_embedding], embeddings)[0]
    except Exception as e:
        print(f"Error en cosine_similarity: {e}")
        return [], []
    
    # Crear lista de resultados combinando similitud semántica y coincidencias textuales
    validated_results = []
    
    for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
        chunk_lower = chunk.lower()
        question_lower = question.lower()

        # Verificar coincidencias textuales específicas
        text_match_score = calculate_text_match_score(question_lower, chunk_lower)

        # Combinar puntuación semántica y textual
        combined_score = similarity * 0.7 + text_match_score * 0.3

        # Umbral más estricto: solo aceptar fragmentos realmente relevantes
        if combined_score >= max(0.55, threshold) and similarity > 0.45:
            validated_results.append((chunk, combined_score, i))
    
    # Si no hay resultados válidos, simplemente no devolver nada
    if not validated_results:
        return [], []
    
    # Ordenar por puntuación combinada descendente
    validated_results.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar solo los mejores resultados
    final_chunks = [result[0] for result in validated_results[:top_k]]
    final_scores = [result[1] for result in validated_results[:top_k]]
    
    return final_chunks, final_scores


def clean_and_summarize(chunk):
    """Limpia y resume el fragmento extraído para mejorar la legibilidad"""
    # Aquí puedes implementar técnicas de limpieza más avanzadas
    cleaned = chunk.replace("\n", " ").strip()
    # Simple summary: truncate long chunks for this demonstration
    return cleaned[:min(200, len(cleaned))] + ('...' if len(cleaned) > 200 else '')


def get_no_answer_response():
    """Genera respuestas variadas y empáticas cuando no se encuentra información"""
    import random
    responses = [
        "🤔 Parece que el documento no aborda ese tema directamente. ¿Te gustaría intentar con otra pregunta relacionada?",
        "😔 No he podido encontrar información específica sobre eso en el texto. ¿Podrías reformular tu pregunta o preguntar sobre otro aspecto?",
        "🔍 Hmm, no encuentro referencias a ese tema en el documento. ¿Hay algo más específico del contenido que te interese?",
        "💭 Lamentablemente, el documento no parece cubrir ese punto. ¿Quizás tengas otra pregunta sobre el material?",
        "🤷‍♂️ No veo que el texto toque ese tema. ¿Te puedo ayudar con alguna otra consulta sobre el documento?"
    ]
    return random.choice(responses)


def get_conversational_template(confidence_level, chunk_count):
    """Obtiene plantillas conversacionales según el nivel de confianza"""
    import random
    
    templates = {
        'high_single': [
            "✨ Perfecto, en el documento se menciona que: {chunk}",
            "📖 Claro, según el texto: {chunk}",
            "🎯 Exacto, el libro explica que: {chunk}",
            "👍 Sí, aquí encontré la respuesta: {chunk}"
        ],
        'high_multiple': [
            "📚 Encontré varias partes relevantes en el documento:",
            "🔍 El texto aborda este tema en diferentes secciones:",
            "📖 Según el documento, hay varias menciones importantes:",
            "✨ He encontrado información completa sobre esto:"
        ],
        'medium': [
            "🤔 Puede que esta parte del documento esté relacionada: {chunk}",
            "💡 Encontré algo que podría ser relevante: {chunk}",
            "🔎 Hay una mención que podría responder tu pregunta: {chunk}",
            "📝 Creo que esto del texto se relaciona con lo que preguntas: {chunk}"
        ]
    }
    
    if confidence_level == 'high' and chunk_count == 1:
        return random.choice(templates['high_single'])
    elif confidence_level == 'high' and chunk_count > 1:
        return random.choice(templates['high_multiple'])
    else:
        return random.choice(templates['medium'])

def extract_most_relevant_part(question, chunk):
    """
    Extrae la frase más relevante y concisa del chunk basándose en la pregunta, priorizando fechas, definiciones y contribuciones.
    """
    if not chunk or not question:
        return ""
    
    import re
    question_lower = question.lower().strip()
    question_words = set(question_lower.split())
    stop_words = {'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para', 'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as', 'al', 'lo', 'le', 'su', 'sus', 'esta', 'este', 'esto', 'esa', 'ese', 'eso'}
    meaningful_words = question_words - stop_words
    
    # Palabras clave para contribuciones
    contrib_keywords = ["contribuci", "aporte", "aportado", "desarrollo", "influencia", "impacto", "relación", "relaciona", "relacionado", "relacionadas"]
    busca_contrib = any(k in question_lower for k in contrib_keywords)
    busca_historia = any(word in question_lower for word in ["historia", "cronolog", "año", "años", "fecha", "origen", "evolución", "inicio", "comienzo"])
    busca_que_es = question_lower.startswith("que es") or question_lower.startswith("¿que es") or question_lower.startswith("qué es") or question_lower.startswith("¿qué es")
    
    # Dividir el chunk en frases usando puntos
    sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        overlap = len(meaningful_words.intersection(sentence_words))
        length_penalty = 1.0 if len(sentence) <= 250 else 0.5
        key_score = overlap * length_penalty
        # Bonus si contiene año o fecha
        if busca_historia and re.search(r'\b(19|20)\d{2}\b', sentence):
            key_score += 2
        # Bonus si parece definición
        if busca_que_es and (sentence_lower.startswith("la ") or sentence_lower.startswith("el ") or sentence_lower.startswith("es ") or sentence_lower.startswith("se define") or "es " in sentence_lower[:20]):
            key_score += 1.5
        # Bonus si contiene palabras de contribución
        if busca_contrib and any(k in sentence_lower for k in contrib_keywords):
            key_score += 2
        # Penalizar encabezados o frases muy cortas
        if len(sentence.split()) < 5 or sentence.isupper():
            key_score *= 0.1
        # Penalizar frases genéricas
        if "contribuciones de las ciencias" in sentence_lower or "varias ciencias han aportado" in sentence_lower:
            key_score *= 0.1
        if key_score > best_score:
            best_score = key_score
            best_sentence = sentence.strip()
    
    # Si no se encontró una frase precisa, devolver el primer enunciado relevante
    if not best_sentence:
        for sentence in sentences:
            if len(sentence.split()) > 5:
                return sentence.strip()[:250] + ("..." if len(sentence) > 250 else "")
        return chunk.strip()[:200] + ("..." if len(chunk) > 200 else "")
    
    return best_sentence[:250] + ("..." if len(best_sentence) > 250 else "")

def generate_response(question, relevant_chunks, relevance_scores=None, intent_info=None):
    """
    Genera una respuesta basada en los fragmentos relevantes encontrados
    """
    if not relevant_chunks or not relevance_scores:
        return "No encontré información sobre esa pregunta en el documento."
    
    best_content = []
    seen_content = set()
    
    # Procesar chunks relevantes
    for chunk, score in zip(relevant_chunks, relevance_scores):
        if chunk in seen_content:
            continue
            
        if score > 0.3:  # Solo usar chunks con buena relevancia
            relevant_part = extract_most_relevant_part(question, chunk)
            if relevant_part and len(relevant_part.strip()) > 10:
                best_content.append(relevant_part)
                seen_content.add(chunk)
                
        if len(best_content) >= 3:  # Limitar a 3 fragmentos más relevantes
            break
    
    # Generar respuesta final
    if best_content:
        # Combinar los mejores fragmentos en una respuesta coherente
        response_parts = []
        for content in best_content:
            if content not in response_parts:  # Evitar duplicados
                response_parts.append(content)
        
        if response_parts:
            return "📖 " + " ".join(response_parts)
    
    return "No encontré información específica sobre esa pregunta en el documento."


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


def is_question_document_related(question, chunks):
    if not chunks:
        return False
    
    question_lower = question.lower().strip()
    # Lista de temas irrelevantes
    unrelated_topics = [
        'tiktok', 'instagram', 'facebook', 'twitter', 'youtube', 'snapchat',
        'capital', 'país', 'ciudad', 'geografía', 'receta', 'cocinar', 'comida',
        'película', 'serie', 'actor', 'cine', 'fútbol', 'soccer', 'deporte',
        'clima', 'tiempo', 'lluvia', 'música', 'canción', 'moda', 'ropa',
        'auto', 'carro', 'vehículo', 'amor', 'pareja', 'novio', 'novia',
        'ecuador', 'colombia', 'perú', 'brasil', 'argentina', 'madrid',
        'barcelona', 'paris', 'london', 'new york', 'uwu', 'lol', 'omg'
    ]
    # Si la pregunta contiene temas claramente no relacionados
    for topic in unrelated_topics:
        if topic in question_lower:
            return False
    
    # Generar embeddings para la pregunta y los chunks
    query_embedding = generate_single_embedding(question)
    doc_embeddings = np.array([generate_single_embedding(chunk) for chunk in chunks[:50]])
    similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    max_score = np.max(similarity_scores)
    # Umbral más estricto: solo considerar relevante si la similitud es alta y hay buen solapamiento de palabras
    if max_score > 0.48:
        # Extraer palabras clave del documento
        document_words = set()
        for chunk in chunks[:10]:
            chunk_words = set(chunk.lower().split())
            document_words.update(chunk_words)
        stop_words = {'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para', 'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the', 'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as'}
        document_words = document_words - stop_words
        question_words = set(question_lower.split()) - stop_words
        if len(question_words) > 0:
            overlap_ratio = len(question_words.intersection(document_words)) / len(question_words)
            if overlap_ratio > 0.22:
                return True
    return False


# Initialize the enhanced search engine globally
_enhanced_search_engine = None

def get_enhanced_search_engine():
    """Get or create the enhanced search engine instance"""
    global _enhanced_search_engine
    if _enhanced_search_engine is None:
        _enhanced_search_engine = EnhancedSearchEngine()
    return _enhanced_search_engine

# Initialize the semantic analyzer globally
_semantic_analyzer = None

def get_semantic_analyzer():
    """Get or create the semantic analyzer instance"""
    global _semantic_analyzer
    if _semantic_analyzer is None:
        _semantic_analyzer = TrueSemanticPDFAnalyzer()
    return _semantic_analyzer


def process_user_query(question, chunks=None, embeddings=None, definition_candidates=None, definition_embeddings=None):
    """
    AI-POWERED CONTEXTUAL PROCESSING PIPELINE
    
    Uses AI to understand questions and provide contextually appropriate responses
    
    Args:
        question (str): Pregunta del usuario
        chunks (list): Lista de fragmentos de texto
        embeddings (np.array): Array de embeddings
        definition_candidates (list): Lista de candidatos a definiciones
        definition_embeddings (np.array): Array de embeddings de definiciones
        
    Returns:
        str: Respuesta final
    """
    # Step 1: Detect intent for non-document questions
    intent_info = detect_intent(question)
    
    # Handle non-document related intents
    if not intent_info.get('requires_document', True):
        return generate_intent_response(intent_info, question)
    
    # Step 2: Check if we have document to analyze
    if not chunks:
        return "📁 Para responder preguntas sobre documentos, primero necesito que subas un archivo PDF."
    
    # Step 3: AI-POWERED QUESTION VALIDATION
    # Check if the question is actually related to the document content
    is_related = ai_validate_question_relevance(question, chunks)
    
    if not is_related:
        return "🤔 Esa pregunta no parece estar relacionada con el contenido del documento. Por favor, pregúntame algo específico sobre el PDF que subiste."
    
    # Step 4: Use Google AI PDF Analyzer for proper understanding and response generation
    try:
        from google_ai_pdf_analyzer import create_google_ai_analyzer
        
        # Create Google AI analyzer
        google_ai_analyzer = create_google_ai_analyzer()
        
        # Load document into Google AI analyzer
        document_text = "\n\n".join(chunks)
        load_result = google_ai_analyzer.load_document(document_text)
        
        if load_result['status'] != 'success':
            # Fallback to enhanced search engine
            return _enhanced_search_fallback(question, chunks)
        
        # Get Google AI-powered response
        ai_result = google_ai_analyzer.answer_question(question)
        
        # Validate the AI response is meaningful and from Google AI
        if (ai_result['confidence'] > 0.3 and 
            ai_result['method'] == 'google_ai_analysis' and 
            not _is_generic_ai_response(ai_result['answer'])):
            return ai_result['answer']
        elif ai_result['method'] == 'unrelated_question':
            # Return the unrelated question response
            return ai_result['answer']
        else:
            # Fallback to enhanced search engine
            return _enhanced_search_fallback(question, chunks)
        
    except Exception as e:
        print(f"Google AI analyzer error: {e}")
        # Fallback to enhanced search engine
        return _enhanced_search_fallback(question, chunks)


def ai_validate_question_relevance(question, chunks):
    """
    Uses AI to validate if a question is actually related to the document content
    
    Args:
        question (str): User's question
        chunks (list): Document chunks
        
    Returns:
        bool: True if question is related to document, False otherwise
    """
    if not chunks or not question:
        return False
    
    question_lower = question.lower().strip()
    
    # Step 1: Check for obviously unrelated topics (be more selective)
    unrelated_topics = [
        'capital', 'país', 'ciudad', 'receta', 'cocinar', 'comida',
        'película', 'serie', 'actor', 'cine', 'fútbol', 'soccer', 'deporte',
        'clima', 'tiempo', 'lluvia', 'música', 'canción', 'moda', 'ropa',
        'auto', 'carro', 'vehículo', 'amor', 'pareja', 'novio', 'novia',
        'ecuador', 'colombia', 'perú', 'brasil', 'argentina', 'madrid',
        'barcelona', 'paris', 'london', 'new york', 'tiktok', 'instagram'
    ]
    
    # Only reject very obviously unrelated topics
    for topic in unrelated_topics:
        if f' {topic} ' in f' {question_lower} ' or question_lower.startswith(topic) or question_lower.endswith(topic):
            return False
    
    # Step 2: Extract document key topics
    document_text = " ".join(chunks[:10])  # Use first 10 chunks for efficiency
    document_lower = document_text.lower()
    
    # Step 3: Look for conceptual overlap
    question_words = set(question_lower.split())
    document_words = set(document_lower.split())
    
    # Remove stop words
    stop_words = {
        'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para',
        'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the',
        'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as'
    }
    
    meaningful_question_words = question_words - stop_words
    meaningful_document_words = document_words - stop_words
    
    if not meaningful_question_words:
        return True  # Allow if question has no meaningful words
    
    # Calculate word overlap
    overlap = len(meaningful_question_words.intersection(meaningful_document_words))
    overlap_ratio = overlap / len(meaningful_question_words)
    
    # Step 4: Use semantic similarity as additional validation
    try:
        question_embedding = generate_single_embedding(question)
        if question_embedding is not None:
            # Sample a few chunks for semantic comparison
            sample_chunks = chunks[:min(5, len(chunks))]
            max_similarity = 0
            
            for chunk in sample_chunks:
                chunk_embedding = generate_single_embedding(chunk)
                if chunk_embedding is not None:
                    similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
                    max_similarity = max(max_similarity, similarity)
            
            # Combined validation: word overlap + semantic similarity
            return (overlap_ratio >= 0.2) or (max_similarity >= 0.45)
    except:
        pass
    
    # Fallback: use word overlap only
    return overlap_ratio >= 0.25


def _is_generic_ai_response(response):
    """
    Check if AI response is generic or contains actual document content
    
    Args:
        response (str): AI generated response
        
    Returns:
        bool: True if response seems generic, False if it contains specific content
    """
    if not response:
        return True
    
    response_lower = response.lower()
    
    # Generic response indicators
    generic_phrases = [
        'no se encuentra en el documento',
        'esta información no se encuentra',
        'no encuentro información',
        'no hay información',
        'el documento no menciona',
        'no está disponible',
        'general', 'key_concepts', 'intent', 'answer_type'  # Mock response artifacts
    ]
    
    for phrase in generic_phrases:
        if phrase in response_lower:
            return True
    
    # Check if response is too short
    if len(response.strip()) < 30:
        return True
    
    return False


def _enhanced_search_fallback(question, chunks):
    """
    Fallback to enhanced search engine when AI contextual chatbot fails
    
    Args:
        question (str): User's question
        chunks (list): Document chunks
        
    Returns:
        str: Response from enhanced search engine
    """
    try:
        # Get the enhanced search engine
        enhanced_engine = get_enhanced_search_engine()
        
        # Reconstruct document text from chunks
        document_text = "\n\n".join(chunks)
        
        # Index the document
        enhanced_engine.index_document(document_text)
        
        # Search using enhanced engine
        enhanced_response = enhanced_engine.search(question)
        
        return enhanced_response
        
    except Exception as e:
        print(f"Enhanced search engine error: {e}")
        # Final fallback to legacy system
        return _legacy_process_query(question, chunks, None, None, None)


def _legacy_process_query(question, chunks, embeddings, definition_candidates, definition_embeddings):
    """
    Legacy processing system as fallback
    """
    # Mejorar la pregunta
    improved_question = improve_question(question)

    # Intentar coincidencia con definiciones primero
    if definition_candidates and definition_embeddings is not None:
        query_embedding = generate_single_embedding(improved_question)
        sim_scores = cosine_similarity([query_embedding], definition_embeddings)[0]
        best_idx = int(np.argmax(sim_scores))
        if sim_scores[best_idx] > 0.7:  # Umbral alto
            return "📖 " + definition_candidates[best_idx]

    # Validación adicional: verificar si la pregunta está realmente relacionada con el documento
    if not is_question_document_related(question, chunks):
        return "🤔 Esa pregunta no parece estar relacionada con el contenido del documento. ¿Podrías preguntarme algo específico sobre el texto que subiste?"

    # 1. Buscar línea/frase exacta en el documento (literal, no plantilla)
    relevant_chunks, scores, specific_lines = analyze_line_by_line(improved_question, chunks)
    if relevant_chunks and specific_lines:
        # Devolver la(s) línea(s) literal(es) más relevante(s) del documento
        # Solo mostrar máximo 2 para evitar respuestas largas
        response_lines = [line for line in specific_lines if line and len(line.strip()) > 10][:2]
        if response_lines:
            return "📖 " + " ".join(response_lines)

    # 2. Si no hay coincidencia literal, usar búsqueda semántica con umbrales graduales
    relevant_chunks, scores = search_similar_chunks(
        improved_question, chunks, embeddings, threshold=0.4  # Umbral inicial moderado
    )
    if not relevant_chunks:
        relevant_chunks, scores = search_similar_chunks(
            improved_question, chunks, embeddings, threshold=0.3
        )
    return generate_response(question, relevant_chunks, scores)
