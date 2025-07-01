"""
Quick test for Novita AI with actual API key
"""

from ai_contextual_chatbot import create_novita_contextual_chatbot

def quick_test():
    print("🚀 Quick Novita AI Test")
    print("=" * 30)
    
    # Use your actual API key
    api_key = "sk_mRJQJG23UKQk4Z6v-p7BOignQM-3e6huQEaqX7ZRfRo"
    model = "meta-llama/llama-3.2-1b-instruct"
    
    print(f"🔑 API Key: {api_key[:20]}...")
    print(f"🤖 Model: {model}")
    
    try:
        # Create chatbot
        print("\n1️⃣ Creating chatbot...")
        chatbot = create_novita_contextual_chatbot(api_key, model)
        print("✅ Chatbot created!")
        
        # Load simple document
        print("\n2️⃣ Loading document...")
        doc = """
        Inteligencia Artificial
        
        La inteligencia artificial (IA) es la capacidad de una máquina para exhibir 
        capacidades cognitivas humanas como el aprendizaje y el razonamiento.
        
        Las principales áreas de la IA incluyen:
        1. Aprendizaje automático
        2. Procesamiento de lenguaje natural  
        3. Visión por computadora
        """
        
        result = chatbot.load_document(doc)
        print(f"📄 Load result: {result['status']}")
        
        if result['status'] == 'success':
            # Test question
            print("\n3️⃣ Testing question...")
            question = "¿Qué es la inteligencia artificial?"
            answer_result = chatbot.answer_question(question)
            
            print(f"❓ Pregunta: {question}")
            print(f"🤖 Respuesta: {answer_result['answer']}")
            print(f"📊 Confianza: {answer_result['confidence']:.2f}")
            
            if "artificial" in answer_result['answer'].lower() or "capacidad" in answer_result['answer'].lower():
                print("\n🎉 ¡SUCCESS! Novita AI integration is working!")
                return True
            else:
                print("\n⚠️ Response seems generic - check API connection")
                
        else:
            print(f"❌ Document load failed: {result['message']}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        if "401" in str(e) or "403" in str(e):
            print("🔑 Authentication error - check API key")
        elif "404" in str(e):
            print("🌐 API endpoint not found")
        else:
            print("💥 Unexpected error")
    
    return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n✅ Integration ready! You can now:")
        print("  • Update your main app.py")
        print("  • Test with real PDF documents")
        print("  • Deploy to production")
    else:
        print("\n🔧 Please check the API key and model configuration")
