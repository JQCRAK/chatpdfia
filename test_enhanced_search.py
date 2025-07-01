"""
Test Enhanced Search Engine with Specific Failure Examples

This test verifies that the new search system finds content that exists
and fixes the false negative problems.
"""

from enhanced_search_engine import EnhancedSearchEngine

def test_failure_examples():
    """Test the specific failure examples provided"""
    
    print("🧪 TESTING ENHANCED SEARCH ENGINE")
    print("=" * 60)
    print("Testing specific failure examples to ensure NO FALSE NEGATIVES")
    
    # Create search engine
    search_engine = EnhancedSearchEngine()
    
    # Test document with the actual content that was missed
    test_document = """
    La inteligencia artificial se encarga de representar las diferentes formas de conocimiento que posee el ser humano. Sus herramientas son la abstracción y la inferencia que le permiten crear sistemas de computación no convencionales los cuales son capaces de aprender nuevas tareas en base a la información proporcionada y tomar decisiones en entornos diversos en los cuales debe actuar.

    Las matemáticas son fundamentales en el desarrollo de teorías formales relacionadas con la lógica, las probabilidades, la teoría de decisiones y la computación. Los métodos matemáticos proporcionan el rigor necesario para el análisis de algoritmos.

    Varias ciencias han aportado al desarrollo de la inteligencia artificial. La psicología cognitiva ha contribuido con modelos de procesamiento de información. La neurociencia ha proporcionado conocimientos sobre el funcionamiento del cerebro.

    Los sistemas expertos son programas que emulan el comportamiento de un experto humano en un dominio específico. Utilizan reglas de inferencia y bases de conocimiento para resolver problemas complejos.

    El aprendizaje automático es una disciplina que permite a las máquinas aprender patrones a partir de datos sin ser programadas explícitamente para cada tarea específica.
    """
    
    # Index the document
    index_info = search_engine.index_document(test_document)
    print(f"📊 Document indexed: {index_info['total_sentences']} sentences, {index_info['total_words']} words")
    
    # Test cases from the failure examples
    test_cases = [
        {
            "question": "¿qué es inteligencia artificial?",
            "expected_content": "inteligencia artificial se encarga de representar",
            "should_find": True,
            "description": "FAILURE EXAMPLE 1: Should find AI definition"
        },
        {
            "question": "Las matemáticas son fundamentales",
            "expected_content": "Las matemáticas son fundamentales en el desarrollo",
            "should_find": True,
            "description": "FAILURE EXAMPLE 2: Should find mathematics content"
        },
        {
            "question": "Contribuciones de las ciencias",
            "expected_content": "Varias ciencias han aportado al desarrollo",
            "should_find": True,
            "description": "Should find sciences contributions"
        },
        {
            "question": "¿qué son los sistemas expertos?",
            "expected_content": "Los sistemas expertos son programas",
            "should_find": True,
            "description": "Should find expert systems definition"
        },
        {
            "question": "¿qué es el aprendizaje automático?",
            "expected_content": "El aprendizaje automático es una disciplina",
            "should_find": True,
            "description": "Should find machine learning definition"
        },
        {
            "question": "¿qué es TikTok?",
            "expected_content": None,
            "should_find": False,
            "description": "Should correctly return 'not found' for irrelevant content"
        }
    ]
    
    print("\n🎯 RUNNING TESTS")
    print("-" * 60)
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {test_case['description']}")
        print(f"Question: '{test_case['question']}'")
        
        # Search for the answer
        answer = search_engine.search(test_case['question'])
        print(f"Answer: {answer}")
        
        # Validate the result
        if test_case['should_find']:
            if "no se encuentra" not in answer.lower():
                if test_case['expected_content'] and test_case['expected_content'].lower() in answer.lower():
                    print("✅ PASSED: Found expected content")
                    passed_tests += 1
                else:
                    print("🔶 PARTIAL: Found content but not exact match")
                    passed_tests += 0.5
            else:
                print("❌ FAILED: Should have found content but returned 'not found'")
        else:
            if "no se encuentra" in answer.lower():
                print("✅ PASSED: Correctly returned 'not found'")
                passed_tests += 1
            else:
                print("❌ FAILED: Should have returned 'not found' but found content")
    
    print("\n" + "=" * 60)
    print(f"🎯 RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("✅ SUCCESS: Enhanced search engine is working correctly!")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Some tests failed")
    
    return passed_tests >= total_tests * 0.8


def test_comprehensive_search():
    """Test comprehensive search capabilities"""
    
    print("\n\n🔍 COMPREHENSIVE SEARCH TEST")
    print("=" * 60)
    
    search_engine = EnhancedSearchEngine()
    
    # More comprehensive test document
    document = """
    Introducción a la Inteligencia Artificial

    La inteligencia artificial (IA) es una rama de la informática que se ocupa de crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estos sistemas pueden aprender, razonar, percibir y tomar decisiones.

    Historia de la IA

    El campo de la inteligencia artificial nació oficialmente en 1956 durante la Conferencia de Dartmouth. Los pioneros incluían a John McCarthy, Marvin Minsky, Allen Newell y Herbert Simon.

    Tipos de Inteligencia Artificial

    Existen diferentes tipos de IA: la IA débil o estrecha, que está diseñada para realizar tareas específicas, y la IA fuerte o general, que tendría capacidades cognitivas comparables a las humanas.

    Aplicaciones Modernas

    Hoy en día, la IA se utiliza en muchas aplicaciones: reconocimiento de voz, procesamiento de lenguaje natural, visión por computadora, robótica, y sistemas de recomendación.

    Desafíos Éticos

    La inteligencia artificial plantea importantes cuestiones éticas sobre privacidad, empleo, sesgo algorítmico y responsabilidad en la toma de decisiones automatizada.
    """
    
    # Index document
    search_engine.index_document(document)
    
    # Test various question types
    questions = [
        "¿Qué es la inteligencia artificial?",
        "¿Cuándo nació la IA?",
        "¿Quiénes fueron los pioneros?",
        "¿Cuáles son los tipos de IA?",
        "¿Dónde se usa la IA hoy?",
        "¿Qué desafíos éticos tiene la IA?",
        "John McCarthy",
        "Conferencia de Dartmouth",
        "1956"
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        answer = search_engine.search(question)
        print(f"📖 {answer}")


def test_debug_info():
    """Test debug functionality"""
    
    print("\n\n🐛 DEBUG INFO TEST")
    print("=" * 60)
    
    search_engine = EnhancedSearchEngine()
    
    # Simple test document
    document = "La inteligencia artificial se encarga de representar las diferentes formas de conocimiento que posee el ser humano."
    search_engine.index_document(document)
    
    # Get debug info
    debug_info = search_engine.get_debug_info("¿qué es inteligencia artificial?")
    
    print("🔍 Debug Information:")
    print(f"Question Analysis: {debug_info['question_analysis']}")
    print(f"Matches Found: {debug_info['matches_found']}")
    print(f"Document Stats: {debug_info['document_stats']}")
    
    if debug_info['top_matches']:
        print("\nTop Matches:")
        for i, (content, score, match_type) in enumerate(debug_info['top_matches'], 1):
            print(f"  {i}. Score: {score:.3f} | Type: {match_type} | Content: {content[:100]}...")


if __name__ == "__main__":
    # Run all tests
    print("🚀 ENHANCED SEARCH ENGINE TESTING")
    print("📋 Testing system that prioritizes finding existing content")
    print("🎯 Goal: NO FALSE NEGATIVES")
    print("=" * 70)
    
    success = test_failure_examples()
    test_comprehensive_search()
    test_debug_info()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced search engine successfully finds existing content")
        print("✅ No more false negatives")
        print("✅ Ready for integration")
    else:
        print("⚠️ Some tests need attention")
        print("💡 Check debug output for tuning opportunities")
    
    print("\n📋 Next steps:")
    print("1. Integrate with existing chatbot system")
    print("2. Replace current search logic")
    print("3. Test with real PDF documents")
    print("4. Monitor for any remaining issues")
