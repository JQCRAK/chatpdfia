"""
Módulo para extraer texto de archivos PDF
"""
import PyPDF2
import streamlit as st


def extract_text_from_pdf(pdf_file):
    """
    Extrae texto de un archivo PDF
    
    Args:
        pdf_file: Archivo PDF subido a través de Streamlit
        
    Returns:
        str: Texto extraído del PDF o None si hay error
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        # Mostrar progreso de extracción
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            
            # Actualizar progreso
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"Procesando página {i + 1} de {total_pages}")
        
        # Limpiar elementos de progreso
        progress_bar.empty()
        status_text.empty()
        
        
        return text
        
    except Exception as e:
        st.error(f"Error al procesar PDF: {str(e)}")
        return None


def validate_pdf_content(text, min_length=100):
    """
    Valida que el texto extraído tenga contenido suficiente
    
    Args:
        text (str): Texto extraído
        min_length (int): Longitud mínima requerida
        
    Returns:
        bool: True si el texto es válido
    """
    if not text or len(text.strip()) < min_length:
        return False
    return True