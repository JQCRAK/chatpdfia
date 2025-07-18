"""
Aplicación principal del chatbot Jahr - Asistente Inteligente para PDFs
"""
import streamlit as st
import time
import os

# Importar módulos locales
from extractor_pdf import extract_text_from_pdf, validate_pdf_content
from fragmentador import create_chunks, get_chunk_statistics
from generador_embeddings import generate_embeddings, validate_embeddings
from motor_busqueda import search_similar_chunks, generate_response, improve_question, process_user_query

# Configuración de la página
st.set_page_config(
    page_title="Jahr - Chatbot PDF",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado mejorado
def load_css():
    """Carga los estilos CSS personalizados"""
    st.markdown("""
    <style>
        /* Variables de color */
        :root {
            --primary-blue: #2563eb;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --success-color: #10b981;
            --error-color: #ef4444;
        }
        
        /* Tema principal */
        .stApp {
            background-color: var(--dark-bg);
            color: var(--text-primary);
        }
        
        /* Header compacto */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, var(--primary-blue), #1d4ed8);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
        }
        
        .main-header h1 {
            font-size: 1.8rem;
            margin: 0;
            font-weight: 600;
        }
        
        .main-header p {
            font-size: 0.9rem;
            margin: 0.25rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Chat container mejorado */
        .chat-container {
            height: 450px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            background: var(--card-bg);
            margin-bottom: 1rem;
        }
        
        /* Mensajes más compactos */
        .user-message {
            background: var(--primary-blue);
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 4px 12px;
            margin: 0.5rem 0 0.5rem 15%;
            max-width: 80%;
            margin-left: auto;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .bot-message {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 12px 4px;
            margin: 0.5rem 15% 0.5rem 0;
            max-width: 80%;
            font-size: 0.9rem;
            line-height: 1.4;
            border-left: 3px solid var(--primary-blue);
        }
        
        /* Estados compactos */
        .status-card {
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            color: #a7f3d0;
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error-color);
            color: #fca5a5;
        }
        
        .status-info {
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid var(--primary-blue);
            color: #93c5fd;
        }
        
        /* Input section más limpia */
        .input-section {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        /* Card de documento activo */
        .document-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .document-card h4 {
            margin: 0 0 0.5rem 0;
            color: var(--success-color);
            font-size: 0.9rem;
        }
        
        .document-info {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.3;
        }
        
        /* Scrollbar personalizado */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: var(--dark-bg);
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        /* Botones más pequeños */
        .stButton > button {
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
            border-radius: 6px;
        }
        
        /* Input más compacto */
        .stTextInput > div > div > input {
            font-size: 0.9rem;
            padding: 0.5rem 0.75rem;
        }
        
        /* Footer compacto */
        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            margin-top: 2rem;
        }
        
        /* Iconos más pequeños en mensajes */
        .message-icon {
            font-size: 0.9rem;
            margin-right: 0.5rem;
        }
        
        /* Upload area mejorada */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
            background: rgba(37, 99, 235, 0.05);
        }
        
        /* Ocultar elementos de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def initialize_session():
    """Inicializa las variables de session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = ""


def display_chat():
    """Muestra el historial de chat"""
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        
        for chat in st.session_state.chat_history:
            if chat['type'] == 'user':
                chat_html += f'<div class="user-message"><span class="message-icon">👤</span>{chat["message"]}</div>'
            else:
                chat_html += f'<div class="bot-message"><span class="message-icon">🤖</span>{chat["message"]}</div>'
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container">
            <div class="bot-message">
                <span class="message-icon">🤖</span>
                <strong>¡Hola! Soy Jahr, tu asistente para documentos PDF.</strong><br><br>
                Sube un documento y podrás preguntarme sobre su contenido.<br><br>
                <strong>Ejemplos:</strong><br>
                • "¿De qué trata este documento?"<br>
                • "Resume los puntos principales"<br>
                • "¿Qué dice sobre [tema específico]?"
            </div>
        </div>
        """, unsafe_allow_html=True)


def process_pdf(uploaded_file):
    """Procesa el archivo PDF subido"""
    with st.spinner("Procesando documento..."):
        # Extraer texto
        text = extract_text_from_pdf(uploaded_file)
        
        if text and validate_pdf_content(text):
            # Crear chunks
            chunks = create_chunks(text)
            
            if chunks:
                # Generar embeddings
                embeddings = generate_embeddings(chunks)
                
                if validate_embeddings(embeddings, len(chunks)):
                    # Guardar en session state
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_processed = True
                    
                    # Estadísticas
                    stats = get_chunk_statistics(chunks)
                    
                    # Mensaje de éxito
                    st.markdown(f"""
                    <div class="status-card status-success">
                        <strong>✅ Documento procesado:</strong> {uploaded_file.name}<br>
                        <strong>Fragmentos:</strong> {stats['total_chunks']} | <strong>Palabras:</strong> ~{stats['total_words']:,}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mensaje de bienvenida
                    welcome_msg = f"Perfecto! He procesado '{uploaded_file.name}' con {len(chunks)} secciones. ¿Qué te gustaría saber?"
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': welcome_msg,
                        'timestamp': time.time()
                    })
                    
                    return True
                else:
                    st.markdown('<div class="status-card status-error">❌ Error al generar embeddings</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-error">❌ No se pudieron extraer fragmentos útiles</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-error">❌ El documento no contiene suficiente texto</div>', unsafe_allow_html=True)
    
    return False


def clear_input():
    """Limpia el campo de entrada usando JavaScript"""
    st.components.v1.html("""
    <script>
        window.parent.document.querySelector('input[aria-label="Mensaje:"]').value = '';
    </script>
    """, height=0)

def handle_user_message(user_input):
    """Procesa el mensaje del usuario y genera respuesta"""
    # Agregar mensaje del usuario
    st.session_state.chat_history.append({
        'type': 'user',
        'message': user_input,
        'timestamp': time.time()
    })
    
    # Procesar consulta de manera integral
    with st.spinner("Generando respuesta..."):
        response = process_user_query(
            user_input,
            st.session_state.chunks if st.session_state.pdf_processed else None,
            st.session_state.embeddings if st.session_state.pdf_processed else None
        )
    
    # Agregar respuesta de Jahr
    st.session_state.chat_history.append({
        'type': 'bot',
        'message': response,
        'timestamp': time.time()
    })

    

def main():
    """Función principal de la aplicación"""
    initialize_session()
    load_css()
    
    # Header compacto
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Jahr</h1>
        <p>Asistente Inteligente para PDFs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout principal
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown("#### 📁 Documento")
        
        # Estado del documento o área de subida
        if st.session_state.pdf_processed:
            st.markdown(f"""
            <div class="document-card">
                <h4>✅ Documento Activo</h4>
                <div class="document-info">
                    <strong>Archivo:</strong> {st.session_state.pdf_name}<br>
                    <strong>Fragmentos:</strong> {len(st.session_state.chunks)}<br>
                    <strong>Estado:</strong> Listo para consultas
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Cambiar documento", type="secondary", use_container_width=True):
                # Reset completo
                for key in ['pdf_processed', 'chunks', 'embeddings', 'chat_history', 'pdf_name']:
                    st.session_state[key] = [] if key in ['chunks', 'embeddings', 'chat_history'] else False if key == 'pdf_processed' else ""
                st.rerun()
        else:
            # Área de subida
            uploaded_file = st.file_uploader(
                "Selecciona tu PDF",
                type="pdf",
                help="Jahr analizará el contenido"
            )
            
            # Procesar PDF
            if uploaded_file is not None:
                if process_pdf(uploaded_file):
                    st.rerun()
    
    with col2:
        st.markdown("#### 💬 Chat con Jahr")
        
        # Mostrar chat
        display_chat()
        
        # Sección de input
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # Input de mensaje con botones en la misma línea
        col_input, col_send, col_clear = st.columns([6, 1.5, 0.8])
        
        with col_input:
            # Usar un key único para cada mensaje
            if 'message_counter' not in st.session_state:
                st.session_state.message_counter = 0
                
            user_input = st.text_input(
                "Mensaje:",
                placeholder="Pregunta sobre el documento o saluda...",
                key=f"user_input_{st.session_state.message_counter}",
                disabled=False,
                label_visibility="collapsed"
            )
        
        with col_send:
            send_clicked = st.button(
                "📤", 
                type="primary", 
                disabled=False,
                use_container_width=True,
                help="Enviar mensaje"
            )
        with col_clear:
            clear_clicked = st.button(
                "🗑️", 
                help="Limpiar chat",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Procesar envío de mensaje
        # Procesar envío de mensaje
        if send_clicked and user_input.strip():
            handle_user_message(user_input)
            # Incrementar contador para crear nuevo input vacío
            st.session_state.message_counter += 1
            st.rerun()
        
        # Detectar Enter key
        if user_input and user_input != st.session_state.get('last_input', ''):
            st.session_state.last_input = user_input
            if len(user_input.strip()) > 0:
                handle_user_message(user_input)
                st.session_state.message_counter += 1
                st.rerun()
        # Limpiar chat
        if clear_clicked:
            if st.session_state.pdf_processed:
                st.session_state.chat_history = [{
                    'type': 'bot',
                    'message': f"Chat reiniciado. ¿Qué más quieres saber sobre '{st.session_state.pdf_name}'?",
                    'timestamp': time.time()
                }]
            else:
                st.session_state.chat_history = []
            st.rerun()
        
        # Mensaje si no hay documento
        if not st.session_state.pdf_processed:
            st.markdown('<div class="status-card status-info">💬 Puedes saludarme o subir un PDF para análisis de documentos</div>', unsafe_allow_html=True)

    # Footer compacto
    st.markdown("""
    <div class="footer">
        🤖 <strong>Jahr v2.0</strong> - Chatbot Inteligente para PDFs
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()