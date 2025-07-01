"""
Test the new AI Semantic PDF Analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_semantic_pdf_analyzer import create_mock_analyzer


def test_sample_pdf_content():
    """Sample PDF content about AI"""
    return """
    Introducción a la Inteligencia Artificial

    La inteligencia es la capacidad de establecer relaciones, las cuales se manifiestan en los seres humanos a través del pensamiento y la parte intelectual, y en los animales de manera puramente sensorial por medio de los sentidos (Artasanchez & Joshi, 2020).

    La inteligencia artificial es una rama de las ciencias de la computación que se enfoca en crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.

    Historia de la Inteligencia Artificial
    
    El desarrollo de la IA comenzó en la década de 1950 con los trabajos pioneros de investigadores como Alan Turing y John McCarthy.

    En 1956, John McCarthy organizó la conferencia de Dartmouth, que es considerada el nacimiento oficial de la inteligencia artificial como campo de estudio.

    En 1977, Feigenbaum desarrolló el sistema EURISKO, pero este fue solo uno de muchos avances en el campo.

    Sistemas Expertos
    
    Sistemas Expertos: Utilizan reglas para representar el conocimiento y la lógica para deducir nuevas informaciones. Estos sistemas son capaces de emular el razonamiento de expertos humanos en dominios específicos.

    Los sistemas expertos son una aplicación práctica de la inteligencia artificial que ha demostrado gran utilidad en diversos campos.

    Áreas de Investigación de la Inteligencia Artificial

    Las principales áreas de investigación en inteligencia artificial incluyen:
    
    1. Manipulación simbólica: Procesamiento de símbolos y representación del conocimiento
    2. Emulación de comportamiento inteligente: Crear sistemas que imiten la inteligencia humana
    3. Aprendizaje automático: Desarrollo de algoritmos que pueden aprender de los datos
    4. Procesamiento de lenguaje natural: Comprensión y generación de lenguaje humano
    5. Visión por computadora: Interpretación de imágenes y videos

    Estas áreas trabajan en conjunto para avanzar el campo de la inteligencia artificial y crear sistemas cada vez más sofisticados.

    Contribuciones de las Ciencias

    Varias ciencias han aportado al desarrollo de la inteligencia artificial, incluyendo las matemáticas, la psicología, la filosofía, y las ciencias de la computación.
    """


def test_ai_semantic_analyzer():
    """Test the AI semantic analyzer with specific questions"""
    
    print("🧪 Testing AI Semantic PDF Analyzer")
    print("=" * 50)
    
    # Create AI analyzer
    analyzer = create_mock_analyzer()
    
    # Load sample document
    sample_content = test_sample_pdf_content()
    load_result = analyzer.load_document(sample_content)
    
    print(f"📄 Document loaded: {load_result}")
    print()
    
    # Test the previously failing questions
    test_questions = [
        ("¿qué es inteligencia?", "Should find the definition from the document"),
        ("Áreas de investigación de la inteligencia artificial", "Should find the research areas list"),
        ("¿qué son Sistemas Expertos?", "Should find the expert systems definition"),
        ("Dame una breve historia cronológica de la inteligencia artificial", "Should find historical information"),
        ("¿qué es espoch?", "Should recognize this is unrelated to the document"),
        ("¿cuál es la capital de Colombia?", "Should recognize this is unrelated to the document")
    ]
    
    print("🔍 Testing Questions:")
    print("-" * 30)
    
    for i, (question, expected) in enumerate(test_questions):
        print(f"\n❓ Question {i+1}: {question}")
        print(f"Expected: {expected}")
        
        # Answer the question
        result = analyzer.answer_question(question)
        
        print(f"📖 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🔧 Method: {result['method']}")
        
        # Analyze if the response is appropriate
        if "no se encuentra" in result['answer'].lower():
            status = "❌ Not found"
        elif "no parece estar relacionada" in result['answer'].lower():
            status = "🚫 Unrelated"
        elif len(result['answer']) > 50 and result['confidence'] > 0.3:
            status = "✅ Good answer"
        else:
            status = "⚠️ Needs improvement"
        
        print(f"Status: {status}")
        print("-" * 30)


if __name__ == "__main__":
    test_ai_semantic_analyzer()
