"""
Test script for the new semantic analysis system
Verifies that TRUE semantic understanding is working correctly
"""

from semantic_analyzer import TrueSemanticPDFAnalyzer

def test_semantic_system():
    """Test the new semantic system with example data"""
    
    # Sample document text that includes definitions
    test_document = """
    Los sistemas de recomendación son herramientas computacionales basadas en algoritmos de aprendizaje profundo diseñadas para sugerir automáticamente productos, servicios, información o acciones a los usuarios.

    La inteligencia artificial es la capacidad de las máquinas de simular procesos cognitivos humanos como el aprendizaje, el razonamiento y la percepción.

    Los sistemas expertos utilizan reglas para representar conocimiento y realizar inferencias. Estos sistemas son especializados en dominios específicos.

    El aprendizaje automático es una rama de la inteligencia artificial que permite a las computadoras aprender patrones de datos sin ser programadas explícitamente.

    Las redes neuronales son estructuras computacionales inspiradas en el cerebro humano que procesan información mediante nodos interconectados.
    """

    # Test questions
    test_questions = [
        "¿Qué son los sistemas de recomendación?",
        "¿Qué es la inteligencia artificial?",
        "¿Qué es TikTok?",  # Should return "not found"
        "¿Cómo funcionan las redes neuronales?",
        "¿Qué es el aprendizaje automático?"
    ]

    # Expected results
    expected_results = [
        "should find recommendation systems definition",
        "should find AI definition", 
        "should return not found",
        "should find neural networks info",
        "should find machine learning definition"
    ]

    # Initialize the semantic analyzer
    analyzer = TrueSemanticPDFAnalyzer()
    
    print("🧪 Testing True Semantic PDF Analyzer")
    print("=" * 50)
    
    for i, question in enumerate(test_questions):
        print(f"\n📝 Test {i+1}: {question}")
        print(f"Expected: {expected_results[i]}")
        
        try:
            result = analyzer.process_document_query(question, test_document)
            print(f"✅ Result: {result}")
            
            # Basic validation
            if "TikTok" in question and "no se encuentra" in result:
                print("✅ Correctly identified irrelevant question")
            elif "TikTok" not in question and "no se encuentra" not in result:
                print("✅ Found relevant information")
            else:
                print("⚠️  Unexpected result")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 Testing complete!")


def test_line_by_line_analysis():
    """Test the line-by-line analysis specifically"""
    from semantic_analyzer import LineByLineDocumentAnalyzer, SemanticQuestionAnalyzer
    
    print("\n🔍 Testing Line-by-Line Analysis")
    print("=" * 50)
    
    # Initialize components
    question_analyzer = SemanticQuestionAnalyzer()
    document_analyzer = LineByLineDocumentAnalyzer()
    
    test_text = """
    Sistemas de Recomendación: Son herramientas computacionales basadas en algoritmos de aprendizaje profundo.
    
    Estos sistemas están diseñados para sugerir automáticamente productos, servicios o información a los usuarios.
    
    La inteligencia artificial permite a las máquinas simular procesos cognitivos humanos.
    
    TikTok es una aplicación de redes sociales.
    """
    
    # Prepare lines
    lines = test_text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    question = "¿Qué son los sistemas de recomendación?"
    
    # Analyze question
    question_analysis = question_analyzer.semantic_question_analysis(question)
    print(f"Question analysis: {question_analysis}")
    
    # Line-by-line search
    relevant_chunks, scores, specific_lines = document_analyzer.line_by_line_document_search(
        question_analysis, lines
    )
    
    print(f"\nRelevant chunks found: {len(relevant_chunks)}")
    for i, (chunk, score) in enumerate(zip(relevant_chunks, scores)):
        print(f"  {i+1}. Score: {score:.3f} | Text: {chunk[:100]}...")


if __name__ == "__main__":
    # Run the tests
    test_semantic_system()
    test_line_by_line_analysis()
