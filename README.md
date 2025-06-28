# Jahr - Chatbot Inteligente para PDFs 🤖

Jahr es un asistente inteligente que te permite hacer preguntas sobre el contenido de documentos PDF usando tecnología de IA avanzada y búsqueda semántica.

## ✨ Características

- **Procesamiento inteligente de PDFs**: Extrae y analiza texto de documentos PDF
- **Búsqueda semántica**: Encuentra información relevante usando similitud de embeddings
- **Interfaz conversacional**: Chat amigable para interactuar con tus documentos
- **Tema oscuro moderno**: Diseño elegante y fácil de usar
- **Respuestas contextuales**: Genera respuestas basadas en el contenido del documento

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

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

4. **Crea la carpeta de datos** (opcional):
```bash
mkdir datos
```

## 🎯 Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

### Cómo usar Jahr

1. **Sube un PDF**: Usa el área de subida en la columna izquierda
2. **Espera el procesamiento**: Jahr analizará el documento automáticamente
3. **Haz preguntas**: Escribe tus preguntas en el chat
4. **Obtén respuestas**: Jahr responderá basándose en el contenido del documento

### Ejemplos de preguntas

- "¿De qué trata este documento?"
- "Resume los puntos principales"
- "¿Qué dice sobre [tema específico]?"
- "Explícame la sección sobre [concepto]"

## 📁 Estructura del Proyecto

```
CHATBOT_IA/
├── app.py                    # Aplicación principal de Streamlit
├── extractor_pdf.py          # Módulo para extraer texto de PDFs
├── fragmentador.py           # Módulo para dividir texto en chunks
├── generador_embeddings.py   # Módulo para generar embeddings
├── motor_busqueda.py         # Módulo de búsqueda semántica
├── requirements.txt          # Dependencias del proyecto
├── README.md                # Documentación del proyecto
└── datos/                   # Carpeta para datos temporales
    ├── texto_extraido.txt   # Texto extraído (opcional)
    └── embeddings_chunks.pkl # Embeddings guardados (opcional)
```

## 🧠 Arquitectura Técnica

### Componentes principales

1. **Extractor de PDF** (`extractor_pdf.py`):
   - Extrae texto de archivos PDF usando PyPDF2
   - Valida el contenido extraído
   - Muestra progreso de extracción

2. **Fragmentador** (`fragmentador.py`):
   - Divide el texto en fragmentos manejables
   - Implementa overlap para mantener contexto
   - Optimiza el tamaño de chunks para embeddings

3. **Generador de Embeddings** (`generador_embeddings.py`):
   - Usa SentenceTransformer para generar embeddings
   - Modelo: `all-MiniLM-L6-v2`
   - Cache para optimizar rendimiento

4. **Motor de Búsqueda** (`motor_busqueda.py`):
   - Búsqueda semántica con similitud coseno
   - Generación de respuestas contextuales
   - Sistema de confianza basado en puntuaciones

### Flujo de procesamiento

```
PDF → Extracción de texto → Fragmentación → Embeddings → Búsqueda → Respuesta
```

## ⚙️ Configuración

### Parámetros personalizables

En `fragmentador.py`:
- `chunk_size`: Tamaño máximo de fragmentos (default: 800)
- `overlap`: Palabras de overlap entre fragmentos (default: 100)

En `motor_busqueda.py`:
- `threshold`: Umbral mínimo de similitud (default: 0.2)
- `top_k`: Número máximo de fragmentos a retornar (default: 3)

### Variables de entorno

Crea un archivo `.env` para configuraciones adicionales:
```
MODEL_NAME=all-MiniLM-L6-v2
MAX_CHUNK_SIZE=800
SIMILARITY_THRESHOLD=0.2
```

## 🔧 Desarrollo

### Estructura modular

El proyecto está diseñado con una arquitectura modular que facilita:
- Mantenimiento del código
- Testing de componentes individuales
- Extensión de funcionalidades
- Reutilización de módulos

### Testing

Para ejecutar tests (si están disponibles):
```bash
pytest tests/
```

### Formateo de código

```bash
black *.py
flake8 *.py
```

## 📊 Rendimiento

### Recomendaciones de hardware

- **RAM mínima**: 4GB
- **RAM recomendada**: 8GB o más
- **CPU**: Cualquier procesador moderno
- **GPU**: Opcional (mejora velocidad de embeddings)

### Limitaciones

- Tamaño máximo de PDF: ~50MB
- Tiempo de procesamiento: Depende del tamaño del documento
- Idiomas soportados: Principalmente español e inglés

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo `LICENSE` para más detalles.

## 🚨 Problemas Comunes

### Error de instalación de PyTorch

Si tienes problemas instalando PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error de memoria

Si el procesamiento falla por memoria:
- Reduce el `chunk_size` en `fragmentador.py`
- Procesa documentos más pequeños
- Aumenta la RAM disponible

### Modelo no se descarga

Si el modelo SentenceTransformer no se descarga:
- Verifica tu conexión a internet
- Intenta con otro modelo más liviano
- Descarga manual del modelo

## 📞 Soporte

Si necesitas ayuda:
1. Revisa la documentación
2. Busca en issues existentes
3. Crea un nuevo issue con detalles del problema

## 🙏 Agradecimientos

- [Streamlit](https://streamlit.io/) por la increíble framework de UI
- [SentenceTransformers](https://www.sbert.net/) por los modelos de embeddings
- [PyPDF2](https://pypdf2.readthedocs.io/) por el procesamiento de PDFs
- [scikit-learn](https://scikit-learn.org/) por las herramientas de ML

---

**Desarrollado con ❤️ para hacer la información más accesible**