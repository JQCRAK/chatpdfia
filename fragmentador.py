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
    # Eliminar saltos de línea múltiples
    text = re.sub(r'\n+', ' ', text)
    return text


def create_chunks(text, chunk_size=800, overlap=100):
    """
    Divide el texto en fragmentos de tamaño específico con overlap
    
    Args:
        text (str): Texto a dividir
        chunk_size (int): Tamaño máximo de cada fragmento
        overlap (int): Palabras de overlap entre fragmentos
        
    Returns:
        list: Lista de fragmentos de texto
    """
    # Limpiar texto
    text = clean_text(text)
    
    # Dividir por párrafos primero
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Si el párrafo es muy largo, dividirlo por oraciones
        if len(paragraph) > chunk_size:
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Mantener overlap
                    words = current_chunk.split()
                    overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk += " " + sentence
        else:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                current_chunk = " ".join(overlap_words) + " " + paragraph
            else:
                current_chunk += " " + paragraph
    
    # Agregar último chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filtrar chunks muy cortos
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
    
    return chunks


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