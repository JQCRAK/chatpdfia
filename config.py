"""
Archivo de configuración para Jahr - Chatbot PDF
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Config:
    """Configuración principal de la aplicación"""
    
    # Configuración del modelo
    MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')
    
    # Configuración de Novita AI
    NOVITA_API_KEY = os.getenv('NOVITA_API_KEY', '')
    NOVITA_MODEL = os.getenv('NOVITA_MODEL', 'meta-llama/llama-3.2-1b-instruct')
    
    # Configuración de fragmentación
    DEFAULT_CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
    DEFAULT_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
    MIN_CHUNK_WORDS = int(os.getenv('MIN_CHUNK_WORDS', '10'))
    
    # Configuración de búsqueda (UMBRALES MUY ESTRICTOS - SOLUCIÓN A PROBLEMAS)
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.75'))  # Incrementado de 0.35 a 0.75
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '2'))  # Reducido de 3 a 2
    
    # Configuración de confianza (UMBRALES ESTRICTOS)
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE', '0.7'))   # Para respuestas seguras
    MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('MEDIUM_CONFIDENCE', '0.6'))  # Umbral medio
    
    # Configuración de validación de relevancia
    WORD_OVERLAP_THRESHOLD = float(os.getenv('WORD_OVERLAP_THRESHOLD', '0.3'))  # 30% palabras en común
    SEMANTIC_VALIDATION_THRESHOLD = float(os.getenv('SEMANTIC_VALIDATION', '0.6'))  # Validación semántica
    
    # Configuración de PDF
    MIN_PDF_LENGTH = int(os.getenv('MIN_PDF_LENGTH', '100'))
    MAX_PDF_SIZE_MB = int(os.getenv('MAX_PDF_SIZE_MB', '50'))
    
    # Configuración de UI
    CHAT_HEIGHT = int(os.getenv('CHAT_HEIGHT', '500'))
    PROGRESS_UPDATE_INTERVAL = float(os.getenv('PROGRESS_INTERVAL', '0.1'))
    
    # Configuración de datos
    DATA_DIR = os.getenv('DATA_DIR', './datos')
    SAVE_EXTRACTED_TEXT = os.getenv('SAVE_TEXT', 'False').lower() == 'true'
    SAVE_EMBEDDINGS = os.getenv('SAVE_EMBEDDINGS', 'False').lower() == 'true'
    
    # Configuración de logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', './logs/jahr.log')
    
    # Información del equipo de creadores
    CREATORS = [
        'Jhostin Quispe',
        'Angel Guaño', 
        'Rumi Grefa',
        'Henry Reding'
    ]
    
    # Mensajes de la aplicación mejorados
    WELCOME_MESSAGE = """
    🤖 ¡Hola! Soy Jahr, tu asistente inteligente para documentos PDF.
    
    📁 Sube un documento y podrás preguntarme cualquier cosa sobre su contenido.
    
    💡 **Ejemplos:**
    • "¿De qué trata este documento?"
    • "Resume los puntos principales"
    • "¿Qué dice sobre [tema específico]?"
    
    💬 **También puedes preguntarme:**
    • "¿Cómo estás?" - Para saludar
    • "¿Cuál es tu nombre?" - Para conocerme
    • "¿Qué puedes hacer?" - Para saber mis capacidades
    • "¿Quién te creó?" - Para conocer a mis creadores
    """
    
    # Respuestas para intenciones conversacionales
    GREETING_RESPONSES = [
        "¡Hola! 👋 Soy Jahr, tu asistente inteligente para documentos PDF. ¿Cómo estás?",
        "¡Hola! 😊 Me alegra saludarte. Estoy aquí para ayudarte con tus documentos PDF.",
        "¡Buenos días! ☀️ Soy Jahr, ¿en qué puedo ayudarte hoy?"
    ]
    
    IDENTITY_RESPONSES = [
        "Soy **Jahr**, tu asistente inteligente especializado en analizar documentos PDF. 🤖\n\nMi función es ayudarte a encontrar información específica en tus documentos y responder preguntas sobre su contenido.",
        "Mi nombre es **Jahr** 🤖 y soy un chatbot diseñado para trabajar con documentos PDF.\n\nPuedo leer, analizar y responder preguntas sobre cualquier PDF que subas."
    ]
    
    CAPABILITIES_RESPONSES = [
        "¡Excelente pregunta! Estas son mis principales capacidades: 🚀\n\n📄 **Análisis de PDFs**: Leo y proceso documentos PDF\n🔍 **Búsqueda inteligente**: Encuentro información específica\n💬 **Respuestas precisas**: Genero respuestas basadas en el documento\n📝 **Resúmenes**: Puedo resumir secciones o todo el documento\n❓ **Preguntas específicas**: Respondo sobre temas particulares del PDF\n\n¡Sube un PDF y podrás preguntarme lo que necesites sobre él!",
        "Puedo ayudarte de muchas formas: 💪\n\n• Analizar documentos PDF completos\n• Responder preguntas específicas sobre el contenido\n• Crear resúmenes de los puntos principales\n• Encontrar información particular que necesites\n• Explicar conceptos complejos del documento\n\n¿Tienes algún PDF que quieras que analice?"
    ]
    
    CREATORS_RESPONSES = [
        f"¡Me enorgullece presentar a mi increíble equipo de creadores! 👨‍💻👩‍💻\n\n🏆 **Mis creadores son:**\n" + 
        "\n".join([f"• **{creator}**" for creator in CREATORS]) +
        "\n\nEste talentoso equipo me diseñó y programó para ser tu mejor asistente PDF. ¡Son geniales! 🌟",
        
        f"Fui creado por un excelente equipo de desarrolladores: 💻✨\n\n👥 **Equipo Jahr:**\n" +
        "\n".join([f"- {creator}" for creator in CREATORS]) +
        "\n\nGracias a ellos puedo ayudarte con tus documentos PDF de manera inteligente."
    ]
    
    THANKS_RESPONSES = [
        "¡De nada! 😊 Estoy aquí para ayudarte siempre que lo necesites.",
        "¡Un placer ayudarte! 🤗 ¿Hay algo más en lo que pueda asistirte?",
        "¡Me alegra haberte sido útil! 😄 No dudes en preguntarme lo que necesites."
    ]
    
    GOODBYE_RESPONSES = [
        "¡Hasta luego! 👋 Fue un placer ayudarte. Vuelve cuando necesites analizar más documentos.",
        "¡Adiós! 😊 Que tengas un excelente día. Aquí estaré cuando regreses.",
        "¡Nos vemos pronto! 🌟 Gracias por usar Jahr."
    ]
    
    ERROR_MESSAGES = {
        'pdf_processing': "❌ Error al procesar el PDF. Verifica que sea un archivo válido.",
        'text_extraction': "❌ No se pudo extraer texto del documento.",
        'insufficient_content': "❌ El documento no contiene suficiente texto.",
        'embedding_generation': "❌ Error al generar embeddings del documento.",
        'search_error': "❌ Error durante la búsqueda. Intenta reformular tu pregunta.",
        'no_results': "🔍 No encontré información específica sobre esa pregunta.",
        'no_document': "📁 Para responder preguntas sobre documentos, primero necesito que subas un archivo PDF."
    }
    
    SUCCESS_MESSAGES = {
        'pdf_processed': "✅ ¡Documento procesado exitosamente!",
        'ready_to_chat': "🎉 Jahr está listo para responder tus preguntas.",
        'chat_cleared': "🔄 Chat reiniciado.",
        'message_sent': "📤 Mensaje enviado correctamente."
    }

    @classmethod
    def create_data_directory(cls):
        """Crea el directorio de datos si no existe"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Valida la configuración"""
        errors = []
        
        if cls.DEFAULT_CHUNK_SIZE < 100:
            errors.append("CHUNK_SIZE debe ser al menos 100")
        
        if cls.SIMILARITY_THRESHOLD < 0 or cls.SIMILARITY_THRESHOLD > 1:
            errors.append("SIMILARITY_THRESHOLD debe estar entre 0 y 1")
        
        if cls.MAX_SEARCH_RESULTS < 1:
            errors.append("MAX_SEARCH_RESULTS debe ser al menos 1")
        
        if errors:
            raise ValueError(f"Errores de configuración: {', '.join(errors)}")
        
        return True

# Configuración específica para desarrollo
class DevConfig(Config):
    """Configuración para desarrollo"""
    LOG_LEVEL = 'DEBUG'
    SAVE_EXTRACTED_TEXT = True
    SAVE_EMBEDDINGS = True

# Configuración específica para producción
class ProdConfig(Config):
    """Configuración para producción"""
    LOG_LEVEL = 'WARNING'
    SAVE_EXTRACTED_TEXT = False
    SAVE_EMBEDDINGS = False

# Seleccionar configuración basada en variable de entorno
ENV = os.getenv('ENVIRONMENT', 'development').lower()

if ENV == 'production':
    config = ProdConfig()
elif ENV == 'development':
    config = DevConfig()
else:
    config = Config()

# Validar configuración al importar
config.validate_config()
config.create_data_directory()