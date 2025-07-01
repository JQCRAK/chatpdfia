"""
Debug test for AI definition
"""
from semantic_analyzer import TrueSemanticPDFAnalyzer, SemanticQuestionAnalyzer, LineByLineDocumentAnalyzer

# Simple test with just AI definition
test_doc = "La inteligencia artificial es la capacidad de las máquinas de simular procesos cognitivos humanos como el aprendizaje, el razonamiento y la percepción."

question = "¿Qué es la inteligencia artificial?"

print("🔍 Debug Test for AI Definition")
print(f"Document: {test_doc}")
print(f"Question: {question}")

# Test each component separately
analyzer = SemanticQuestionAnalyzer()
doc_analyzer = LineByLineDocumentAnalyzer() 

# Analyze the question
question_analysis = analyzer.semantic_question_analysis(question)
print(f"\nQuestion Analysis: {question_analysis}")

# Prepare document lines
lines = [test_doc]

# Test line-by-line search
relevant_chunks, scores, specific_lines = doc_analyzer.line_by_line_document_search(
    question_analysis, lines
)

print(f"\nLine-by-line results:")
print(f"Relevant chunks: {len(relevant_chunks)}")
for i, (chunk, score) in enumerate(zip(relevant_chunks, scores)):
    print(f"  {i+1}. Score: {score:.3f} | Text: {chunk}")

# Test full system
full_analyzer = TrueSemanticPDFAnalyzer()
result = full_analyzer.process_document_query(question, test_doc)
print(f"\nFull system result: {result}")
