"""
Módulo para dividir texto en fragmentos (chunks)
"""
import re


def clean_text(text):
    """
    Limpia el texto eliminando espacios excesivos y caracteres innecesarios
    
    Args:
        text (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text.strip())
    # Eliminar saltos de línea múltiples pero mantener separación de párrafos
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def is_header_or_metadata(text):
    """
    Detecta si un fragmento es principalmente encabezado, título o metadatos
    
    Args:
        text (str): Fragmento de texto a evaluar
        
    Returns:
        bool: True si es encabezado/metadatos
    """
    text_lower = text.lower().strip()
    
    # Patrones de encabezados y metadatos
    header_patterns = [
        r'^(capítulo|chapter|parte|part)\s*\d+',
        r'^\d+\.\s*(introducción|introduction)',
        r'isbn.*\d',
        r'volumen\s*\d+',
        r'volume\s*\d+',
        r'^fundamentals of',
        r'^tabla de contenido',
        r'^table of contents',
        r'^índice',
        r'^index',
        r'^bibliografía',
        r'^bibliography',
        r'^referencias',
        r'^references'
    ]
    
    # Verificar patrones
    for pattern in header_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Si el texto es muy corto y tiene muchas mayúsculas, probablemente es título
    if len(text) < 100:
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if uppercase_ratio > 0.3:
            return True
    
    # Si tiene muchos números y pocos verbos, probablemente es metadatos
    words = text.split()
    if len(words) < 20:
        number_ratio = sum(1 for word in words if any(c.isdigit() for c in word)) / max(len(words), 1)
        if number_ratio > 0.4:
            return True
    
    return False


def has_content_indicators(text):
    """
    Verifica si un fragmento tiene indicadores de contenido sustancial
    
    Args:
        text (str): Fragmento de texto
        
    Returns:
        bool: True si tiene contenido sustancial
    """
    text_lower = text.lower()
    
    # Indicadores de contenido sustancial
    content_indicators = [
        ':', 'son', 'es', 'se define', 'significa', 'consiste', 'incluye',
        'permite', 'utiliza', 'funciona', 'caracteriza', 'representa',
        'ejemplo', 'como', 'mediante', 'través', 'proceso', 'método',
        'sistema', 'algoritmo', 'técnica', 'approach', 'method', 'system'
    ]
    
    indicator_count = sum(1 for indicator in content_indicators if indicator in text_lower)
    
    # Si tiene definiciones o explicaciones
    if ':' in text and any(word in text_lower for word in ['son', 'es', 'se define', 'significa']):
        return True
    
    # Si tiene múltiples indicadores de contenido
    if indicator_count >= 2:
        return True
    
    return False


def extract_definition_candidates(text):
    """
    Extract substantive content that answers questions, avoiding keywords/metadata
    """
    import re
    candidates = []
    
    # Split into paragraphs for better context
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or len(paragraph.split()) < 8:
            continue
            
        # Skip obvious metadata/keywords sections - but be more specific
        paragraph_lower = paragraph.lower()
        if (paragraph_lower.startswith('keywords:') or 
            paragraph_lower.startswith('palabras clave:') or 
            paragraph_lower.startswith('key words:') or
            'isbn' in paragraph_lower or
            paragraph_lower.startswith('volumen') or
            paragraph_lower.startswith('capítulo') or
            len(paragraph.split()) < 10):  # Skip very short paragraphs
            continue
            
        # Further filter out sections with repetitious or irrelevant patterns
        if paragraph.lower().startswith('capítulo') or paragraph.isupper():
            continue
            
        # Skip headers and titles (too short, mostly uppercase)
        if len(paragraph) < 100 or paragraph.isupper():
            continue
            
        # Look for substantive explanatory content
        explanatory_patterns = [
            r'La importancia de [^.]+\.',
            r'[A-ZÁÉÍÓÚ][^:]{2,80}:\s[^\n]{20,}',  # Definitions with substantial content
            r'[^.]+consiste en [^.]{20,}\.',
            r'[^.]+se debe a [^.]{20,}\.',
            r'[^.]+radica en [^.]{20,}\.',
            r'[^.]+permite [^.]{20,}\.',
            r'[^.]+ayuda a [^.]{20,}\.',
            r'[^.]+contribuye a [^.]{20,}\.',
        ]
        
        # Check if paragraph contains substantive explanatory content
        for pattern in explanatory_patterns:
            if re.search(pattern, paragraph, re.IGNORECASE):
                if not is_header_or_metadata(paragraph):
                    candidates.append(paragraph)
                    break
        
        # Also include paragraphs that start with topics and have substantial explanation
        if (':' in paragraph and 
            len(paragraph) > 150 and  # Must be substantial
            any(indicator in paragraph.lower() for indicator in ['permite', 'ayuda', 'contribuye', 'consiste', 'incluye', 'significa']) and
            not is_header_or_metadata(paragraph)):
            candidates.append(paragraph)
    
    return candidates

def extract_semantic_units(text):
    """
    Extrae unidades semánticamente coherentes del texto (definiciones, conceptos, explicaciones)
    
    Args:
        text (str): Texto completo
        
    Returns:
        list: Lista de unidades semánticas
    """
    semantic_units = []
    
    # Patrón para encontrar definiciones
    definition_patterns = [
        r'([A-Z][^:]+):\s*([^.!?]+[.!?])',  # "Concepto: definición."
        r'([A-Z][^:]+):\s*([^\n]+)',        # "Concepto: definición"
        r'(\w+\s+\w+)\s+son\s+([^.]+\.)',  # "Los sistemas son..."
        r'(\w+\s+\w+)\s+es\s+([^.]+\.)',   # "La inteligencia es..."
        r'Se\s+define\s+(\w+)\s+como\s+([^.]+\.)' # "Se define AI como..."
    ]
    
    for pattern in definition_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            concept = match.group(1).strip()
            definition = match.group(2).strip()
            if len(definition) > 20 and not is_header_or_metadata(f"{concept}: {definition}"):
                semantic_units.append({
                    'type': 'definition',
                    'concept': concept,
                    'content': f"{concept}: {definition}",
                    'score': 1.0
                })
    
    return semantic_units


def create_concept_preserving_chunks(text, max_size=600):
    """
    Crea chunks preservando conceptos y definiciones completas
    
    Args:
        text (str): Texto a fragmentar
        max_size (int): Tamaño máximo de chunk
        
    Returns:
        list: Lista de chunks semánticamente coherentes
    """
    # Primero extraer unidades semánticas
    semantic_units = extract_semantic_units(text)
    
    # Dividir en párrafos
    paragraphs = text.split('\n\n')
    concept_chunks = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or len(paragraph) < 30:
            continue
            
        if is_header_or_metadata(paragraph):
            continue
            
        # Si el párrafo contiene definiciones, preservarlo completo si es posible
        if ':' in paragraph and any(indicator in paragraph.lower() for indicator in ['son', 'es', 'se define', 'significa']):
            if len(paragraph) <= max_size:
                concept_chunks.append({
                    'content': paragraph,
                    'type': 'definition_paragraph',
                    'score': 0.9
                })
                continue
        
        # Para párrafos largos, dividir por oraciones pero mantener coherencia
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + " " + sentence) <= max_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    concept_chunks.append({
                        'content': current_chunk,
                        'type': 'paragraph_chunk',
                        'score': 0.7
                    })
                current_chunk = sentence
        
        if current_chunk:
            concept_chunks.append({
                'content': current_chunk,
                'type': 'paragraph_chunk',
                'score': 0.7
            })
    
    # Combinar con unidades semánticas extraídas
    all_units = semantic_units + concept_chunks
    
    # Eliminar duplicados y ordenar por relevancia
    unique_chunks = []
    seen_content = set()
    
    for unit in sorted(all_units, key=lambda x: x['score'], reverse=True):
        content = unit['content']
        # Verificar si ya tenemos contenido similar
        is_duplicate = False
        for seen in seen_content:
            if len(set(content.lower().split()) & set(seen.lower().split())) > len(content.split()) * 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate and len(content.split()) > 5:
            unique_chunks.append(content)
            seen_content.add(content)
    
    return unique_chunks


def create_chunks(text, chunk_size=600, overlap=50, min_chunk_length=50):
    """
    Crea chunks preservando la coherencia semántica y las definiciones completas
    Implementa filtrado estricto para evitar fragmentos sin sentido
    
    Args:
        text (str): Texto a dividir
        chunk_size (int): Tamaño máximo de cada fragmento
        overlap (int): Solapamiento entre fragmentos
        min_chunk_length (int): Longitud mínima de cada chunk (caracteres)
        
    Returns:
        list: Lista de fragmentos semánticamente coherentes
    """
    # Limpiar texto
    text = clean_text(text)
    
    # Crear chunks preservando conceptos
    concept_chunks = create_concept_preserving_chunks(text, chunk_size)
    
    # Si no hay suficientes chunks conceptuales, usar método tradicional como respaldo
    if len(concept_chunks) < 3:
        # Método tradicional mejorado
        paragraphs = text.split('\n\n')
        traditional_chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            # Filtrado estricto: mínimo 50 caracteres y no metadatos
            if not paragraph or len(paragraph) < min_chunk_length or is_header_or_metadata(paragraph):
                continue
                
            if len(paragraph) <= chunk_size:
                traditional_chunks.append(paragraph)
            else:
                # Dividir párrafo largo manteniendo coherencia
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Verificar si agregar la oración excede el tamaño
                    if len(current_chunk + " " + sentence) <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        # Guardar chunk actual si cumple requisitos mínimos
                        if current_chunk and len(current_chunk) >= min_chunk_length:
                            traditional_chunks.append(current_chunk)
                        current_chunk = sentence
                
                # Guardar último chunk si cumple requisitos
                if current_chunk and len(current_chunk) >= min_chunk_length:
                    traditional_chunks.append(current_chunk)
        
        # Combinar chunks conceptuales y tradicionales
        concept_chunks.extend(traditional_chunks)
    
    # Filtrado final estricto para calidad
    final_chunks = []
    for chunk in concept_chunks:
        # More refined filtering for chunks
        chunk_lower = chunk.lower().strip()
        if (len(chunk) >= min_chunk_length and  # Mínimo 50 caracteres
            len(chunk.split()) >= 8 and  # Mínimo 8 palabras
            not is_header_or_metadata(chunk) and  # No metadatos
            not chunk_lower.startswith('keywords:') and  # No empiece con keywords
            not chunk_lower.startswith('palabras clave:') and  # No empiece con palabras clave
            not chunk_lower.startswith('key words:') and  # No empiece con key words
            not chunk_lower.startswith('capítulo') and  # No capítulos
            not chunk_lower.startswith('chapter') and  # No chapters
            not chunk_lower.startswith('isbn') and  # No ISBN
            not (len(chunk) < 200 and chunk_lower.count('keywords') > 0)):  # Avoid short keyword lists
            
            # Verificar que tenga contenido sustancial
            if has_content_indicators(chunk) or len(chunk) > 100:
                final_chunks.append(chunk)
    
    return final_chunks


def get_chunk_statistics(chunks):
    """
    Obtiene estadísticas de los fragmentos creados
    
    Args:
        chunks (list): Lista de fragmentos
        
    Returns:
        dict: Diccionario con estadísticas
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'total_words': 0
        }
    
    lengths = [len(chunk) for chunk in chunks]
    word_counts = [len(chunk.split()) for chunk in chunks]
    
    return {
        'total_chunks': len(chunks),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'total_words': sum(word_counts)
    }