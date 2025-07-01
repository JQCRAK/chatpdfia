"""
Google AI Studio PDF Analyzer

This module uses Google's Gemini AI model to analyze PDF documents and answer questions 
based ONLY on the content found in the document. The AI is instructed to:
1. Read and understand the PDF content
2. Analyze user questions
3. Provide accurate answers using only document information
4. Reject questions that cannot be answered from the document
"""

import os
import re
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
import time


class GoogleAIPDFAnalyzer:
    """Google AI Studio implementation for PDF document analysis and Q&A"""
    
    def __init__(self, api_key: str = "AIzaSyCp7_hlZCPq_FPMl9zQS8N0D09yxrjaqqw"):
        """Initialize Google AI with the provided API key"""
        self.api_key = api_key
        self.model = None
        self.document_content = ""
        self.document_loaded = False
        
        try:
            # Configure Google AI
            genai.configure(api_key=self.api_key)
            
            # Initialize the Gemini model (updated to latest model name)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            print("✅ Google AI Studio configured successfully")
            
        except Exception as e:
            print(f"❌ Error configuring Google AI: {e}")
            self.model = None
    
    def load_document(self, pdf_text: str) -> Dict:
        """Load PDF document for analysis"""
        if not pdf_text or len(pdf_text.strip()) < 100:
            return {
                'status': 'error',
                'message': 'Document is too short or empty'
            }
        
        try:
            self.document_content = pdf_text.strip()
            self.document_loaded = True
            
            # Get document statistics
            word_count = len(self.document_content.split())
            char_count = len(self.document_content)
            
            return {
                'status': 'success',
                'message': f'Document loaded successfully',
                'word_count': word_count,
                'char_count': char_count
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error loading document: {str(e)}'
            }
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using Google AI based on the loaded PDF document"""
        if not self.document_loaded:
            return {
                'answer': '📁 Por favor, sube un documento PDF primero.',
                'confidence': 0.0,
                'method': 'no_document'
            }
        
        if not self.model:
            return {
                'answer': '⚠️ Error: Google AI no está configurado correctamente.',
                'confidence': 0.0,
                'method': 'ai_error'
            }
        
        if not question or len(question.strip()) < 3:
            return {
                'answer': '❓ Por favor, haz una pregunta más específica.',
                'confidence': 0.0,
                'method': 'invalid_question'
            }
        
        try:
            # Step 1: Check if question is related to document using AI
            relevance_check = self._check_question_relevance(question)
            
            if not relevance_check['is_related']:
                return {
                    'answer': '🤔 Esa pregunta no parece estar relacionada con el contenido del documento. Por favor, pregúntame algo específico sobre el PDF que subiste.',
                    'confidence': 0.0,
                    'method': 'unrelated_question'
                }
            
            # Step 2: Generate answer using AI with strict document-only instructions
            ai_response = self._generate_document_based_answer(question)
            
            # Step 3: Validate the response
            validation_result = self._validate_response(question, ai_response)
            
            if validation_result['is_valid']:
                return {
                    'answer': ai_response,
                    'confidence': validation_result['confidence'],
                    'method': 'google_ai_analysis'
                }
            else:
                return {
                    'answer': 'No pude encontrar información específica sobre esa pregunta en el documento.',
                    'confidence': 0.0,
                    'method': 'validation_failed'
                }
        
        except Exception as e:
            print(f"Error in Google AI analysis: {e}")
            return {
                'answer': 'Hubo un error procesando tu pregunta. Por favor, intenta reformularla.',
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _check_question_relevance(self, question: str) -> Dict:
        """Use Google AI to check if the question is related to the document"""
        # First do a quick fallback check to be more permissive
        fallback_result = self._fallback_relevance_check(question)
        
        # If fallback says it's related, trust it (be more permissive)
        if fallback_result['is_related']:
            return fallback_result
        
        # Only use AI for borderline cases
        relevance_prompt = f"""
Analiza si esta pregunta puede responderse con el contenido del siguiente documento:

PREGUNTA: {question}

DOCUMENTO:
{self.document_content[:2000]}...

Instrucciones:
1. Si la pregunta menciona conceptos, temas o términos que aparecen en el documento, responde "SÍ"
2. Si la pregunta es sobre temas académicos, científicos o técnicos relacionados con el documento, responde "SÍ" 
3. Solo responde "NO" si la pregunta es claramente sobre temas completamente diferentes (geografía, entretenimiento, deportes, etc.)
4. En caso de duda, responde "SÍ"

Respuesta (solo SÍ o NO):"""

        try:
            response = self.model.generate_content(
                relevance_prompt,
                generation_config={
                    'max_output_tokens': 10,
                    'temperature': 0.1
                }
            )
            
            response_text = response.text.strip().upper()
            is_related = "SÍ" in response_text or "SI" in response_text or "YES" in response_text
            
            return {
                'is_related': is_related,
                'confidence': 0.7 if is_related else 0.3
            }
            
        except Exception as e:
            print(f"Error checking question relevance: {e}")
            # Default to allowing the question (be permissive)
            return {'is_related': True, 'confidence': 0.6}
    
    def _generate_document_based_answer(self, question: str) -> str:
        """Generate answer using Google AI with strict document-only instructions"""
        
        answer_prompt = f"""
Eres un asistente especializado en analizar documentos PDF. Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contenido del documento proporcionado.

REGLAS CRÍTICAS:
1. USA SOLO la información que aparece en el documento
2. NO agregues información de tu conocimiento general
3. Si el documento no contiene la respuesta, di "Esta información no se encuentra en el documento"
4. Sé preciso y directo
5. Usa un lenguaje natural y claro
6. Cita partes específicas del documento cuando sea relevante

PREGUNTA DEL USUARIO: {question}

CONTENIDO COMPLETO DEL DOCUMENTO:
{self.document_content}

INSTRUCCIONES FINALES:
- Lee cuidadosamente todo el documento
- Busca la información específica que responde a la pregunta
- Proporciona una respuesta completa y precisa usando solo el contenido del documento
- Si la información no está en el documento, responde exactamente: "Esta información no se encuentra en el documento"

RESPUESTA:"""

        try:
            response = self.model.generate_content(
                answer_prompt,
                generation_config={
                    'max_output_tokens': 500,
                    'temperature': 0.1
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "Error al generar la respuesta. Por favor, intenta de nuevo."
    
    def _validate_response(self, question: str, response: str) -> Dict:
        """Validate that the AI response is appropriate and based on document content"""
        if not response or len(response.strip()) < 10:
            return {'is_valid': False, 'confidence': 0.0}
        
        response_lower = response.lower()
        
        # Check for "not found" responses
        if any(phrase in response_lower for phrase in [
            'no se encuentra en el documento',
            'esta información no se encuentra',
            'no está en el documento',
            'el documento no menciona'
        ]):
            return {'is_valid': True, 'confidence': 0.9}
        
        # Check for hallucination indicators
        hallucination_phrases = [
            'en general', 'típicamente', 'comúnmente', 'es bien sabido',
            'los estudios muestran', 'según los expertos', 'es conocido que',
            'in general', 'typically', 'commonly', 'it is known that'
        ]
        
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                return {'is_valid': False, 'confidence': 0.0}
        
        # Check if response contains content that could be from the document
        document_words = set(self.document_content.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le',
            'da', 'su', 'por', 'son', 'con', 'para', 'una', 'del', 'las', 'los', 'al', 'como',
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        meaningful_response_words = response_words - stop_words
        meaningful_document_words = document_words - stop_words
        
        if meaningful_response_words:
            overlap = len(meaningful_response_words.intersection(meaningful_document_words))
            overlap_ratio = overlap / len(meaningful_response_words)
            
            # High overlap suggests response is based on document content
            if overlap_ratio >= 0.6:
                confidence = min(0.9, 0.6 + overlap_ratio * 0.3)
                return {'is_valid': True, 'confidence': confidence}
            elif overlap_ratio >= 0.3:
                confidence = min(0.7, 0.3 + overlap_ratio * 0.4)
                return {'is_valid': True, 'confidence': confidence}
            else:
                return {'is_valid': False, 'confidence': 0.1}
        
        return {'is_valid': True, 'confidence': 0.5}
    
    def _fallback_relevance_check(self, question: str) -> Dict:
        """Fallback method to check question relevance without AI"""
        question_lower = question.lower()
        
        # Check for obviously unrelated topics
        unrelated_topics = [
            'capital', 'país', 'ciudad', 'geografía', 'receta', 'cocinar', 'comida',
            'película', 'serie', 'actor', 'cine', 'fútbol', 'deporte', 'clima',
            'música', 'canción', 'moda', 'ropa', 'auto', 'carro', 'amor', 'pareja',
            'ecuador', 'colombia', 'perú', 'brasil', 'madrid', 'barcelona', 'paris',
            'tiktok', 'instagram', 'facebook', 'espoch'
        ]
        
        for topic in unrelated_topics:
            if topic in question_lower:
                return {'is_related': False, 'confidence': 0.8}
        
        # Simple word overlap check
        document_words = set(self.document_content.lower().split())
        question_words = set(question_lower.split())
        
        stop_words = {'qué', 'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una'}
        meaningful_question_words = question_words - stop_words
        
        if meaningful_question_words:
            overlap = len(meaningful_question_words.intersection(document_words))
            overlap_ratio = overlap / len(meaningful_question_words)
            
            return {
                'is_related': overlap_ratio >= 0.2,
                'confidence': 0.7 if overlap_ratio >= 0.2 else 0.3
            }
        
        return {'is_related': True, 'confidence': 0.5}
    
    def get_document_summary(self) -> str:
        """Generate a summary of the loaded document using Google AI"""
        if not self.document_loaded or not self.model:
            return "No hay documento cargado o AI no está disponible."
        
        summary_prompt = f"""
Crea un resumen conciso del siguiente documento en español:

DOCUMENTO:
{self.document_content}

Instrucciones:
1. Resume los puntos principales del documento
2. Mantén el resumen entre 100-200 palabras
3. Usa solo la información del documento
4. Escribe en español claro y natural

RESUMEN:"""

        try:
            response = self.model.generate_content(
                summary_prompt,
                generation_config={
                    'max_output_tokens': 300,
                    'temperature': 0.2
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"Error generando resumen: {e}"


# Factory function to create the analyzer
def create_google_ai_analyzer() -> GoogleAIPDFAnalyzer:
    """Create a Google AI PDF analyzer instance"""
    return GoogleAIPDFAnalyzer()


# Test function
def test_google_ai_analyzer():
    """Test the Google AI analyzer with sample content"""
    analyzer = create_google_ai_analyzer()
    
    # Sample document content
    sample_content = """
    Introducción a la Inteligencia Artificial

    La inteligencia es la capacidad de establecer relaciones, las cuales se manifiestan en los seres humanos a través del pensamiento y la parte intelectual, y en los animales de manera puramente sensorial por medio de los sentidos (Artasanchez & Joshi, 2020).

    La inteligencia artificial es una rama de las ciencias de la computación que se enfoca en crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.

    Historia de la Inteligencia Artificial
    
    El desarrollo de la IA comenzó en la década de 1950 con los trabajos pioneros de investigadores como Alan Turing y John McCarthy. En 1956, John McCarthy organizó la conferencia de Dartmouth, que es considerada el nacimiento oficial de la inteligencia artificial como campo de estudio.

    Sistemas Expertos
    
    Sistemas Expertos: Utilizan reglas para representar el conocimiento y la lógica para deducir nuevas informaciones. Estos sistemas son capaces de emular el razonamiento de expertos humanos en dominios específicos.

    Áreas de Investigación de la Inteligencia Artificial

    Las principales áreas de investigación en inteligencia artificial incluyen:
    1. Manipulación simbólica: Procesamiento de símbolos y representación del conocimiento
    2. Emulación de comportamiento inteligente: Crear sistemas que imiten la inteligencia humana
    3. Aprendizaje automático: Desarrollo de algoritmos que pueden aprender de los datos
    4. Procesamiento de lenguaje natural: Comprensión y generación de lenguaje humano
    5. Visión por computadora: Interpretación de imágenes y videos
    """
    
    # Load document
    load_result = analyzer.load_document(sample_content)
    print(f"Document load result: {load_result}")
    
    # Test questions
    test_questions = [
        "¿qué es inteligencia?",
        "¿Cuáles son las áreas de investigación de la IA?",
        "¿qué son los Sistemas Expertos?",
        "Dame una breve historia de la IA",
        "¿cuál es la capital de Colombia?",  # Should be rejected
        "¿qué es espoch?"  # Should be rejected
    ]
    
    print("\n🔍 Testing Questions:")
    print("-" * 50)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = analyzer.answer_question(question)
        print(f"📖 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🔧 Method: {result['method']}")
        print("-" * 30)


if __name__ == "__main__":
    test_google_ai_analyzer()
