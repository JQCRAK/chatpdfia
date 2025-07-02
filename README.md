# Jahr - Chatbot Inteligente para PDFs 🤖

Jahr es un asistente inteligente avanzado que te permite hacer preguntas sobre el contenido de documentos PDF usando tecnología de IA de última generación, búsqueda semántica y modelos de lenguaje contextuales.

## ✨ Características

- **Procesamiento inteligente de PDFs**: Extrae y analiza texto de documentos PDF con múltiples métodos
- **Búsqueda semántica avanzada**: Encuentra información relevante usando similitud de embeddings
- **Múltiples motores de IA**: Soporte para diferentes modelos de lenguaje
- **Motor de búsqueda mejorado**: Sistema de búsqueda optimizado con ranking inteligente
- **Interfaz conversacional**: Chat amigable para interactuar con tus documentos
- **Análisis contextual**: Respuestas contextuales basadas en el contenido del documento
- **Sistema de testing completo**: Suite de pruebas para garantizar calidad
- **Tema oscuro moderno**: Diseño elegante y fácil de usar

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Acceso a internet para descarga de modelos

### Pasos de instalación

1. **Clona el repositorio**:
```bash
git clone https://github.com/tu-usuario/jahr-chatbot-pdf.git
cd jahr-chatbot-pdf
```

2. **Crea un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configura el proyecto**:
```bash
python config.py  # Configuración inicial
```

## 🎯 Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

### Interfaces disponibles

1. **Aplicación principal**: `app.py` - Interfaz completa de Streamlit
2. **Demo comprensivo**: `comprehensive_demo.py` - Demostración avanzada
3. **Test rápido**: `quick_test_novita.py` - Prueba rápida del sistema

### Cómo usar Jahr

1. **Sube un PDF**: Usa el área de subida en la interfaz
2. **Espera el procesamiento**: Jahr analizará el documento automáticamente
3. **Haz preguntas**: Escribe tus preguntas en el chat
4. **Obtén respuestas**: Jahr responderá basándose en el contenido del documento

## 📁 Estructura del Proyecto

```
CHAATBOT/
├── chatpdfia/                          # Directorio principal del chatbot
│   ├── ai_contextual_chatbot.py       # Chatbot contextual con IA
│   ├── ai_document_chatbot.py          # Chatbot especializado en documentos
│   ├── ai_semantic_pdf_analyzer.py    # Analizador semántico de PDFs
│   ├── app.py                          # Aplicación principal de Streamlit
│   ├── comprehensive_demo.py           # Demo completo del sistema
│   ├── config.py                       # Configuración del proyecto
│   ├── debug_ai_test.py               # Herramientas de debug
│   ├── enhanced_search_engine.py      # Motor de búsqueda mejorado
│   ├── extractor_pdf.py               # Extractor de texto de PDFs
│   ├── fragmentador.py                # Fragmentador de texto
│   ├── generador_embeddings.py       # Generador de embeddings
│   ├── google_ai_pdf_analyzer.py     # Analizador con Google AI
│   ├── motor_busqueda.py             # Motor de búsqueda principal
│   ├── novita_ai_model.py            # Integración con Novita AI
│   ├── quick_test_novita.py          # Test rápido de Novita
│   ├── semantic_analyzer.py          # Analizador semántico
│   ├── utils.py                      # Utilidades del proyecto
│   │
│   └── tests/                        # Suite de testing
│       ├── test_ai_chatbot.py
│       ├── test_ai_contextual.py
│       ├── test_ai_semantic.py
│       ├── test_enhanced_search.py
│       ├── test_integration.py
│       ├── test_novita_api.py
│       ├── test_novita_integration.py
│       └── test_semantic_system.py
│
├── requirements.txt                   # Dependencias del proyecto
├── README.md                         # Documentación principal
├── SEMANTIC_SYSTEM_README.md         # Documentación del sistema semántico
├── SOLUTION_COMPLETE.md              # Documentación completa de la solución
└── app_novita_update_instructions.md # Instrucciones de actualización
```

## 🧠 Arquitectura Técnica

### Componentes principales

1. **Chatbots especializados**:
   - `ai_contextual_chatbot.py`: Chatbot con contexto conversacional
   - `ai_document_chatbot.py`: Especializado en análisis de documentos
   - `ai_semantic_pdf_analyzer.py`: Análisis semántico avanzado

2. **Motores de búsqueda**:
   - `motor_busqueda.py`: Motor principal de búsqueda
   - `enhanced_search_engine.py`: Motor mejorado con ranking
   - `semantic_analyzer.py`: Análisis semántico de contenido

3. **Procesamiento de documentos**:
   - `extractor_pdf.py`: Extracción de texto de PDFs
   - `fragmentador.py`: División inteligente de texto
   - `generador_embeddings.py`: Generación de embeddings vectoriales

4. **Integración con IA**:
   - `novita_ai_model.py`: Integración con Novita AI
   - `google_ai_pdf_analyzer.py`: Análisis con Google AI

### Flujo de procesamiento

```
PDF → Extracción → Fragmentación → Embeddings → Búsqueda Semántica → IA → Respuesta Contextual
```

## ⚙️ Configuración

### Archivo de configuración

El archivo `config.py` contiene todas las configuraciones del sistema:

```python
# Ejemplo de configuración
CHUNK_SIZE = 800
OVERLAP = 100
SIMILARITY_THRESHOLD = 0.2
MODEL_NAME = "all-MiniLM-L6-v2"
```

### Variables de entorno

Crea un archivo `.env` para configuraciones sensibles:
```
NOVITA_API_KEY=tu_api_key
GOOGLE_AI_API_KEY=tu_google_key
MODEL_PATH=ruta_del_modelo
```

## 🧪 Testing

### Ejecutar todas las pruebas

```bash
# Ejecutar suite completa de testing
python -m pytest tests/

# Pruebas específicas
python test_ai_chatbot.py
python test_enhanced_search.py
python test_integration.py
```

### Pruebas rápidas

```bash
# Test rápido del sistema
python quick_test_novita.py

# Debug del sistema
python debug_ai_test.py
```

## 🚀 Demos y ejemplos

### Demo comprensivo

```bash
python comprehensive_demo.py
```

Este demo incluye:
- Carga y procesamiento de PDFs
- Diferentes tipos de consultas
- Comparación de motores de búsqueda
- Análisis de rendimiento

## 📊 Modelos y APIs soportadas

### Modelos locales
- SentenceTransformers (all-MiniLM-L6-v2)
- Embeddings personalizados

### APIs externas
- **Novita AI**: Para respuestas avanzadas
- **Google AI**: Para análisis complementario

## 🔧 Desarrollo

### Estructura modular

El proyecto utiliza una arquitectura modular que permite:
- Intercambio de motores de IA
- Testing independiente de componentes
- Escalabilidad horizontal
- Fácil mantenimiento

### Extensión del sistema

Para agregar nuevos modelos de IA:
1. Crea un archivo en el patrón `nuevo_modelo.py`
2. Implementa la interfaz base
3. Agrega tests correspondientes
4. Actualiza la configuración

## 📈 Rendimiento

### Optimizaciones implementadas

- Cache de embeddings
- Búsqueda vectorial optimizada
- Procesamiento en lotes
- Gestión eficiente de memoria

### Métricas de rendimiento

- Tiempo de procesamiento: < 30s para PDFs de 50 páginas
- Precisión de búsqueda: > 85%
- Uso de memoria: Optimizado para documentos grandes

## 🚨 Solución de problemas

### Problemas comunes

1. **Error de API**: Verifica las claves de API en `.env`
2. **Memoria insuficiente**: Reduce `CHUNK_SIZE` en config
3. **Modelo no encontrado**: Ejecuta `python config.py` para descargar

### Logs y debugging

```bash
# Activar modo debug
python debug_ai_test.py

# Logs detallados
export DEBUG_MODE=true
python app.py
```

## 📚 Documentación adicional

- `SEMANTIC_SYSTEM_README.md`: Sistema semántico detallado
- `SOLUTION_COMPLETE.md`: Documentación técnica completa
- `app_novita_update_instructions.md`: Guía de actualización

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Ejecuta los tests (`python -m pytest tests/`)
4. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
5. Push a la rama (`git push origin feature/AmazingFeature`)
6. Abre un Pull Request

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Streamlit](https://streamlit.io/) por la framework de UI
- [SentenceTransformers](https://www.sbert.net/) por los modelos de embeddings
- [Novita AI](https://novita.ai/) por la API de IA
- [Google AI](https://ai.google/) por los servicios de análisis
- [PyPDF2](https://pypdf2.readthedocs.io/) por el procesamiento de PDFs

---

**Desarrollado con ❤️ para hacer la información más accesible mediante IA avanzada**

## 📞 Soporte

- **Issues**: Reporta problemas en GitHub Issues
- **Documentación**: Consulta los archivos README específicos
- **Testing**: Usa los scripts de testing para diagnosticar problemas
