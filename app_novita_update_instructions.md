# Updating app.py for Novita AI Integration

## Instructions for adding Novita AI to your Streamlit app

### 1. Update imports in app.py

Add this import at the top of your app.py file:

```python
from ai_contextual_chatbot import create_novita_contextual_chatbot
```

### 2. Add Novita AI option to model selection

Find the model selection part in your app.py and add Novita AI as an option:

```python
# In your sidebar or model selection area
ai_model_option = st.selectbox(
    "🤖 Selecciona el modelo de IA:",
    ["Mock (Testing)", "OpenAI GPT", "Claude", "Gemini", "Novita AI"],
    index=0
)
```

### 3. Add Novita AI configuration

Add Novita AI configuration in your sidebar:

```python
if ai_model_option == "Novita AI":
    st.sidebar.subheader("⚡ Configuración Novita AI")
    novita_api_key = st.sidebar.text_input(
        "API Key de Novita AI:",
        type="password",
        value=Config.NOVITA_API_KEY,
        help="Ingresa tu API key de Novita AI"
    )
    novita_model = st.sidebar.selectbox(
        "Modelo Novita:",
        ["novita/meta-llama/llama-3.2-1b-instruct"],
        index=0
    )
    
    if novita_api_key:
        st.sidebar.success("✅ API Key configurada")
    else:
        st.sidebar.warning("⚠️ Necesitas una API Key de Novita AI")
```

### 4. Update chatbot initialization

Find where you initialize the chatbot and add the Novita AI case:

```python
# Initialize chatbot based on selected model
if ai_model_option == "Mock (Testing)":
    st.session_state.chatbot = create_mock_contextual_chatbot()
elif ai_model_option == "OpenAI GPT":
    if openai_api_key:
        st.session_state.chatbot = create_openai_contextual_chatbot(openai_api_key, openai_model)
    else:
        st.error("🔑 Necesitas proporcionar una API Key de OpenAI")
        st.session_state.chatbot = None
elif ai_model_option == "Claude":
    if claude_api_key:
        st.session_state.chatbot = create_claude_contextual_chatbot(claude_api_key, claude_model)
    else:
        st.error("🔑 Necesitas proporcionar una API Key de Claude")
        st.session_state.chatbot = None
elif ai_model_option == "Gemini":
    if gemini_api_key:
        st.session_state.chatbot = create_gemini_contextual_chatbot(gemini_api_key, gemini_model)
    else:
        st.error("🔑 Necesitas proporcionar una API Key de Gemini")
        st.session_state.chatbot = None
elif ai_model_option == "Novita AI":
    if novita_api_key:
        st.session_state.chatbot = create_novita_contextual_chatbot(novita_api_key, novita_model)
        st.success("🚀 Chatbot Novita AI inicializado")
    else:
        st.error("🔑 Necesitas proporcionar una API Key de Novita AI")
        st.session_state.chatbot = None
```

### 5. Add model information display

Add information about the currently selected model:

```python
# Display current model info
if ai_model_option == "Novita AI" and novita_api_key:
    st.info(f"🤖 Usando Novita AI: {novita_model}")
```

### 6. Add error handling for Novita AI

Add specific error handling for Novita AI API issues:

```python
# In your answer processing section
try:
    result = st.session_state.chatbot.answer_question(user_question)
    # ... existing code ...
except Exception as e:
    error_msg = str(e)
    if "401" in error_msg or "403" in error_msg:
        st.error("🔑 Error de autenticación con Novita AI. Verifica tu API key.")
    elif "novita" in error_msg.lower():
        st.error(f"⚡ Error de Novita AI: {error_msg}")
    else:
        st.error(f"❌ Error: {error_msg}")
```

### 7. Environment Variable Setup

Create or update your `.env` file to include:

```bash
# Novita AI Configuration
NOVITA_API_KEY=your_novita_api_key_here
NOVITA_MODEL=novita/meta-llama/llama-3.2-1b-instruct
```

### 8. Test the Integration

1. Install the new requirements:
```bash
pip install -r requirements.txt
```

2. Run the test script:
```bash
python test_novita_integration.py
```

3. Start your Streamlit app:
```bash
streamlit run app.py
```

4. Select "Novita AI" from the model dropdown and test with a PDF document.

## Features Added

✅ **Novita AI Model Support**: Full integration with Novita's Llama model  
✅ **API Key Configuration**: Secure API key input in sidebar  
✅ **Error Handling**: Specific error messages for Novita AI issues  
✅ **Model Selection**: Easy switching between different AI models  
✅ **Environment Variables**: Support for .env configuration  

## Troubleshooting

### Common Issues:

1. **API Key Issues**: 
   - Verify the API key is correct
   - Check if you have credits/access to the model
   - Try creating a new API key

2. **Model Access**: 
   - Ensure you have access to the specific Llama model
   - Check Novita AI documentation for available models

3. **Network Issues**:
   - Verify internet connection
   - Check if Novita AI service is available

### Getting Help:

- Check the test script output for specific error messages
- Review Novita AI documentation
- Contact your development team for API key assistance

## Next Steps

Once the API key issue is resolved:

1. ✅ Test with the provided test script
2. ✅ Update your main app.py with the changes above
3. ✅ Test with real PDF documents
4. ✅ Deploy to production

Your Novita AI integration is now ready! 🎉
