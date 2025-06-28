"""
Módulo para generar embeddings de texto usando SentenceTransformer
"""
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    """
    Carga el modelo de SentenceTransformer con cache para optimizar rendimiento
    
    Returns:
        SentenceTransformer: Modelo cargado
    """
    return SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(chunks):
    """
    Genera embeddings para una lista de fragmentos de texto
    
    Args:
        chunks (list): Lista de fragmentos de texto
        
    Returns:
        np.array: Array de embeddings
    """
    if not chunks:
        return np.array([])
    
    model = load_model()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = model.encode([chunk])[0]
            embeddings.append(embedding)
            
            # Actualizar progreso
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            status_text.text(f"Generando embeddings: {i + 1}/{len(chunks)}")
            
        except Exception as e:
            st.error(f"Error generando embedding para chunk {i}: {str(e)}")
            continue
    
    # Limpiar elementos de progreso
    progress_bar.empty()
    status_text.empty()
    
    return np.array(embeddings)


def generate_single_embedding(text):
    """
    Genera un embedding para un texto individual
    
    Args:
        text (str): Texto para el cual generar embedding
        
    Returns:
        np.array: Embedding del texto
    """
    model = load_model()
    try:
        embedding = model.encode([text])[0]
        return embedding
    except Exception as e:
        st.error(f"Error generando embedding: {str(e)}")
        return None


def validate_embeddings(embeddings, expected_count):
    """
    Valida que los embeddings se hayan generado correctamente
    
    Args:
        embeddings (np.array): Array de embeddings
        expected_count (int): Número esperado de embeddings
        
    Returns:
        bool: True si los embeddings son válidos
    """
    if embeddings is None or len(embeddings) == 0:
        return False
    
    if len(embeddings) != expected_count:
        st.warning(f"Advertencia: Se esperaban {expected_count} embeddings, pero se generaron {len(embeddings)}")
    
    # Verificar que no haya embeddings nulos o inválidos
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        st.error("Error: Se encontraron valores inválidos en los embeddings")
        return False
    
    return True