"""
Test script for Novita AI integration with Jahr Chatbot
Run this to test the complete Novita AI integration
"""

from ai_contextual_chatbot import create_novita_contextual_chatbot
from config import Config
import os

def test_novita_chatbot():
    """Test the complete Novita AI chatbot integration"""
    
    print("🚀 Testing Novita AI Integration with Jahr Chatbot")
    print("=" * 60)
    
    # Get API key - you can also set it in environment variables
    api_key = "sk_mRJQJG23UKQk4Z6v-p7BOignQM-3e6huQEaqX7ZRfRo"
    
    # Or use from config/environment
    if Config.NOVITA_API_KEY:
        api_key = Config.NOVITA_API_KEY
        print(f"📋 Using API key from config: {api_key[:20]}...")
    else:
        print(f"📋 Using provided API key: {api_key[:20]}...")
    
    try:
        # Create Novita AI chatbot
        print("\n1️⃣ Creating Novita AI chatbot...")
        chatbot = create_novita_contextual_chatbot(api_key)
        print("✅ Chatbot created successfully!")
        
        # Test with sample document
        print("\n2️⃣ Testing with sample document...")
        sample_document = """
        Inteligencia Artificial
        
        La inteligencia artificial (IA) es la capacidad de una máquina para exhibir capacidades cognitivas humanas, 
        como el aprendizaje, el razonamiento y la autocorrección.
        
        Áreas de investigación en IA:
        
        1. Aprendizaje automático (Machine Learning)
        2. Procesamiento de lenguaje natural
        3. Visión por computadora
        4. Robótica
        5. Sistemas expertos
        
        Los sistemas expertos utilizan bases de conocimiento para resolver problemas específicos 
        en dominios particulares, como diagnóstico médico o análisis financiero.
        """
        
        # Load document
        load_result = chatbot.load_document(sample_document)
        print(f"📄 Document load result: {load_result}")
        
        if load_result['status'] == 'success':
            print("✅ Document loaded successfully!")
            
            # Test questions
            test_questions = [
                "¿Qué es la inteligencia artificial?",
                "¿Cuáles son las áreas de investigación en IA?",
                "¿Qué son los sistemas expertos?"
            ]
            
            print("\n3️⃣ Testing questions with Novita AI...")
            for i, question in enumerate(test_questions, 1):
                print(f"\n❓ Pregunta {i}: {question}")
                try:
                    result = chatbot.answer_question(question)
                    print(f"🤖 Respuesta: {result['answer']}")
                    print(f"📊 Confianza: {result['confidence']:.2f}")
                    print(f"🔍 Info de procesamiento: {result['processing_info']}")
                except Exception as e:
                    print(f"❌ Error answering question: {str(e)}")
            
            print("\n4️⃣ Testing debug information...")
            try:
                debug_info = chatbot.get_debug_info(test_questions[0])
                print("🔧 Debug info:")
                for key, value in debug_info.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"❌ Error getting debug info: {str(e)}")
                
        else:
            print(f"❌ Failed to load document: {load_result['message']}")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        
        # Check if it's an API key issue
        if "401" in str(e) or "403" in str(e) or "auth" in str(e).lower():
            print("\n💡 API Key Issue Detected!")
            print("🔑 To fix this:")
            print("1. Verify your Novita AI API key is correct")
            print("2. Check if the API key is active/funded")
            print("3. Ensure you have access to the specified model")
            print("4. Try creating a new API key from your Novita AI dashboard")
        else:
            print(f"\n💡 Error details: {str(e)}")

def setup_environment_variables():
    """Setup environment variables for easier testing"""
    print("\n🔧 Environment Setup Instructions:")
    print("To avoid hardcoding API keys, create a .env file with:")
    print("NOVITA_API_KEY=your_actual_api_key_here")
    print("NOVITA_MODEL=novita/meta-llama/llama-3.2-1b-instruct")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found!")
    else:
        print("⚠️  .env file not found. Creating template...")
        with open('.env', 'w') as f:
            f.write("# Novita AI Configuration\n")
            f.write("NOVITA_API_KEY=sk-your-api-key-here\n")
            f.write("NOVITA_MODEL=novita/meta-llama/llama-3.2-1b-instruct\n")
        print("📝 Created .env template file. Please add your actual API key.")

def integration_checklist():
    """Show integration checklist"""
    print("\n📋 Novita AI Integration Checklist:")
    print("=" * 50)
    
    checklist_items = [
        ("✅", "novita_ai_model.py created"),
        ("✅", "ai_contextual_chatbot.py updated with Novita support"),
        ("✅", "config.py updated with Novita configuration"),
        ("✅", "requirements.txt updated with requests dependency"),
        ("⚠️", "API key validation (needs working key)"),
        ("⚠️", "End-to-end testing (pending API key fix)")
    ]
    
    for status, item in checklist_items:
        print(f"{status} {item}")
    
    print("\n🎯 Next Steps:")
    print("1. Get a valid Novita AI API key")
    print("2. Run this test script again")
    print("3. Update your main app.py to use Novita AI")
    print("4. Test with real PDF documents")

if __name__ == "__main__":
    print("🔬 Novita AI Integration Test Suite")
    print("=" * 50)
    
    # Run tests
    test_novita_chatbot()
    
    # Show setup instructions
    setup_environment_variables()
    
    # Show integration checklist
    integration_checklist()
    
    print("\n🎉 Integration test complete!")
    print("📞 Contact your team if you need help with API key setup.")
