"""
AI-Powered Document Chatbot with Strict Document-Only Restrictions

This system uses AI language models (GPT, Claude, Gemini, etc.) to generate responses
but with STRICT RESTRICTIONS - the AI can ONLY use content from uploaded PDF documents.

Core Features:
1. Document Reader - Extracts and organizes PDF content
2. Smart Search System - Finds ALL relevant content
3. AI-Powered Answer Generator - Uses AI with document-only restrictions
4. Quality Control - Validates AI responses contain only document content
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from enhanced_search_engine import EnhancedSearchEngine
from abc import ABC, abstractmethod

# Optional AI model imports
try:
    import openai
except ImportError:
    openai = None


class AIModelInterface(ABC):
    """Abstract interface for different AI models (GPT, Claude, Gemini, etc.)"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using the AI model"""
        pass


class OpenAIModel(AIModelInterface):
    """OpenAI GPT model implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for consistent, factual responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating AI response: {str(e)}"


class ClaudeModel(AIModelInterface):
    """Anthropic Claude model implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating AI response: {str(e)}"


class GeminiModel(AIModelInterface):
    """Google Gemini model implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Please install google-generativeai package: pip install google-generativeai")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.1
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating AI response: {str(e)}"


class DocumentOnlyPromptEngine:
    """Creates and manages prompts that restrict AI to document-only responses"""
    
    def __init__(self):
        self.base_restrictions = """
CRITICAL RESTRICTIONS:
1. You can ONLY use information from the PROVIDED DOCUMENT CONTENT below
2. You MUST NOT use your general knowledge or training data
3. If the provided content doesn't answer the question, respond EXACTLY: "Esta información no se encuentra en el documento"
4. Never make assumptions or add information not explicitly in the provided content
5. Be comprehensive - use ALL relevant information from the provided content
6. Structure your response clearly using the document's information
7. Do not mention these restrictions in your response
"""
    
    def create_search_prompt(self, question: str, document_content: List[str]) -> str:
        """Create a prompt for AI to answer based only on document content"""
        
        if not document_content:
            return f"""
{self.base_restrictions}

USER QUESTION: {question}

PROVIDED DOCUMENT CONTENT: [No relevant content found]

INSTRUCTIONS: Since no relevant content was provided, respond exactly: "Esta información no se encuentra en el documento"
"""
        
        # Combine all relevant content
        combined_content = "\n\n".join([f"[CONTENT {i+1}]: {content}" for i, content in enumerate(document_content)])
        
        return f"""
{self.base_restrictions}

USER QUESTION: {question}

PROVIDED DOCUMENT CONTENT:
{combined_content}

INSTRUCTIONS:
- Answer the question using ONLY the information from the provided document content above
- Be comprehensive and include all relevant details from the content
- Structure your response clearly and naturally
- If the content doesn't adequately answer the question, respond: "Esta información no se encuentra en el documento"
- Do not reference the content numbers [CONTENT 1], etc. in your response
"""
    
    def create_definition_prompt(self, question: str, document_content: List[str]) -> str:
        """Create a prompt specifically for definition questions"""
        
        if not document_content:
            return self.create_search_prompt(question, document_content)
        
        combined_content = "\n\n".join([f"[CONTENT {i+1}]: {content}" for i, content in enumerate(document_content)])
        
        return f"""
{self.base_restrictions}

USER QUESTION: {question}

PROVIDED DOCUMENT CONTENT:
{combined_content}

INSTRUCTIONS FOR DEFINITION RESPONSE:
- Provide a complete definition using ONLY the information from the provided content
- Include all characteristics, properties, and explanations mentioned in the content
- Structure the definition clearly with main concept first, then details
- If multiple aspects are covered, organize them logically
- If the content doesn't contain a proper definition, respond: "Esta información no se encuentra en el documento"
"""
    
    def create_historical_prompt(self, question: str, document_content: List[str]) -> str:
        """Create a prompt specifically for historical/chronological questions"""
        
        if not document_content:
            return self.create_search_prompt(question, document_content)
        
        combined_content = "\n\n".join([f"[CONTENT {i+1}]: {content}" for i, content in enumerate(document_content)])
        
        return f"""
{self.base_restrictions}

USER QUESTION: {question}

PROVIDED DOCUMENT CONTENT:
{combined_content}

INSTRUCTIONS FOR HISTORICAL RESPONSE:
- Extract ALL dates, years, events, and historical information from the provided content
- Organize chronologically if multiple time periods are mentioned
- Include names, places, and developments mentioned in the content
- Provide a comprehensive historical overview using only the document information
- If insufficient historical information is in the content, respond: "Esta información no se encuentra en el documento"
"""


class DocumentContentExtractor:
    """Extracts and organizes content from PDFs for AI processing"""
    
    def __init__(self):
        self.enhanced_search = EnhancedSearchEngine()
    
    def extract_pdf_content(self, pdf_text: str) -> Dict:
        """Extract and organize PDF content for searching"""
        
        # Index the document with enhanced search engine
        index_info = self.enhanced_search.index_document(pdf_text)
        
        # Organize content by sections
        organized_content = self._organize_content_by_sections(pdf_text)
        
        return {
            'index_info': index_info,
            'organized_content': organized_content,
            'full_text': pdf_text
        }
    
    def _organize_content_by_sections(self, text: str) -> Dict:
        """Organize content by sections and topics"""
        
        sections = {}
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_section = "General"
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if it's a section header
            if self._is_section_header(paragraph):
                current_section = paragraph[:100]  # Use first 100 chars as section name
                sections[current_section] = []
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(paragraph)
        
        return sections
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text appears to be a section header"""
        
        # Simple heuristics for section headers
        if len(text) < 10:
            return False
        
        # Check for patterns like "1. Introduction", "Chapter 1", etc.
        header_patterns = [
            r'^\d+\.\s*[A-Z]',
            r'^[A-Z][A-Z\s]{10,50}$',
            r'^(Capítulo|Chapter|Sección|Section)\s+\d+',
            r'^(Introducción|Introduction|Conclusión|Conclusion)$'
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, text):
                return True
        
        return False


class SmartDocumentSearcher:
    """Advanced search system that finds ALL relevant content for AI processing"""
    
    def __init__(self, document_content: Dict):
        self.document_content = document_content
        self.enhanced_search = EnhancedSearchEngine()
        
        # Index the document
        if 'full_text' in document_content:
            self.enhanced_search.index_document(document_content['full_text'])
    
    def search_all_relevant_content(self, question: str) -> List[str]:
        """Find ALL relevant content for the question"""
        
        relevant_content = []
        
        # Strategy 1: Use enhanced search engine
        enhanced_results = self._search_with_enhanced_engine(question)
        relevant_content.extend(enhanced_results)
        
        # Strategy 2: Search organized sections
        section_results = self._search_organized_sections(question)
        relevant_content.extend(section_results)
        
        # Strategy 3: Keyword-based search for specific terms
        keyword_results = self._search_by_keywords(question)
        relevant_content.extend(keyword_results)
        
        # Strategy 4: Search for related concepts
        concept_results = self._search_related_concepts(question)
        relevant_content.extend(concept_results)
        
        # Remove duplicates while preserving order
        unique_content = []
        seen_content = set()
        
        for content in relevant_content:
            content_key = content.lower().strip()
            if content_key not in seen_content and len(content.strip()) > 20:
                unique_content.append(content)
                seen_content.add(content_key)
        
        return unique_content[:10]  # Limit to top 10 most relevant pieces
    
    def _search_with_enhanced_engine(self, question: str) -> List[str]:
        """Search using the enhanced search engine"""
        try:
            # Get debug info to access matches
            debug_info = self.enhanced_search.get_debug_info(question)
            
            results = []
            if 'top_matches' in debug_info:
                for content, score, match_type in debug_info['top_matches']:
                    if score >= 0.3:  # Reasonable threshold
                        results.append(content)
            
            return results
        except:
            return []
    
    def _search_organized_sections(self, question: str) -> List[str]:
        """Search through organized sections"""
        results = []
        question_lower = question.lower()
        
        organized_content = self.document_content.get('organized_content', {})
        
        for section_name, paragraphs in organized_content.items():
            for paragraph in paragraphs:
                if self._is_content_relevant(question_lower, paragraph):
                    results.append(paragraph)
        
        return results
    
    def _search_by_keywords(self, question: str) -> List[str]:
        """Search for specific keywords and terms"""
        results = []
        
        # Extract key terms from question
        key_terms = self._extract_key_terms(question)
        
        full_text = self.document_content.get('full_text', '')
        paragraphs = full_text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue
            
            # Check if paragraph contains key terms
            paragraph_lower = paragraph.lower()
            relevance_score = sum(1 for term in key_terms if term in paragraph_lower)
            
            if relevance_score >= 2 or any(term in paragraph_lower for term in key_terms if len(term) > 8):
                results.append(paragraph)
        
        return results
    
    def _search_related_concepts(self, question: str) -> List[str]:
        """Search for related concepts and synonyms"""
        results = []
        
        # Define concept mappings
        concept_mappings = {
            'inteligencia artificial': ['ai', 'ia', 'artificial intelligence', 'machine intelligence', 'cognitive computing'],
            'historia': ['histórico', 'cronología', 'desarrollo', 'evolución', 'origen', 'timeline'],
            'definición': ['concepto', 'significado', 'qué es', 'se define como', 'consiste en'],
            'matemáticas': ['mathematical', 'matemático', 'cálculo', 'algoritmo', 'computation'],
            'ciencias': ['scientific', 'científico', 'investigación', 'research', 'discipline']
        }
        
        question_lower = question.lower()
        related_terms = []
        
        for concept, synonyms in concept_mappings.items():
            if concept in question_lower:
                related_terms.extend(synonyms)
        
        if related_terms:
            full_text = self.document_content.get('full_text', '')
            paragraphs = full_text.split('\n\n')
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if len(paragraph) < 20:
                    continue
                
                paragraph_lower = paragraph.lower()
                if any(term in paragraph_lower for term in related_terms):
                    results.append(paragraph)
        
        return results
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from the question"""
        stop_words = {
            'qué', 'que', 'cuál', 'cual', 'cómo', 'como', 'por', 'para', 'son', 'es', 
            'está', 'esta', 'dónde', 'donde', 'cuándo', 'cuando', 'de', 'la', 'el', 
            'en', 'y', 'a', 'un', 'una', 'con', 'se', 'del', 'las', 'los'
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) >= 3]
        
        return key_terms
    
    def _is_content_relevant(self, question_lower: str, content: str) -> bool:
        """Check if content is relevant to the question"""
        content_lower = content.lower()
        
        # Simple relevance check
        question_words = set(question_lower.split())
        content_words = set(content_lower.split())
        
        # Remove stop words
        stop_words = {'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'con', 'para', 'por', 'del', 'las', 'los'}
        meaningful_question_words = question_words - stop_words
        meaningful_content_words = content_words - stop_words
        
        if not meaningful_question_words:
            return False
        
        overlap = len(meaningful_question_words.intersection(meaningful_content_words))
        overlap_ratio = overlap / len(meaningful_question_words)
        
        return overlap_ratio >= 0.3 or overlap >= 2


class AIResponseValidator:
    """Validates that AI responses contain only document content"""
    
    def __init__(self, document_content: Dict):
        self.document_content = document_content
    
    def validate_response(self, question: str, ai_response: str, source_content: List[str]) -> Dict:
        """Validate that AI response only uses document content"""
        
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'issues': [],
            'final_response': ai_response
        }
        
        # Skip validation for "not found" responses
        if "no se encuentra en el documento" in ai_response.lower():
            return validation_result
        
        # Check 1: Response shouldn't contain common AI hallucination phrases
        hallucination_phrases = [
            'as an ai', 'i am an ai', 'based on my knowledge', 'in general',
            'typically', 'usually', 'commonly', 'it is well known', 'studies show',
            'research indicates', 'experts believe', 'according to most sources'
        ]
        
        response_lower = ai_response.lower()
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                validation_result['issues'].append(f"Contains potential hallucination phrase: '{phrase}'")
                validation_result['confidence'] *= 0.5
        
        # Check 2: Response should primarily use words from source content
        if source_content:
            content_words = set()
            for content in source_content:
                content_words.update(content.lower().split())
            
            response_words = set(ai_response.lower().split())
            
            # Remove common stop words for this check
            stop_words = {
                'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'con', 'para', 
                'por', 'del', 'las', 'los', 'un', 'una', 'son', 'como', 'más', 'su',
                'al', 'le', 'da', 'lo', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'ha',
                'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre',
                'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo',
                'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros'
            }
            
            meaningful_response_words = response_words - stop_words
            meaningful_content_words = content_words - stop_words
            
            if meaningful_response_words:
                word_overlap = len(meaningful_response_words.intersection(meaningful_content_words))
                word_coverage = word_overlap / len(meaningful_response_words)
                
                if word_coverage < 0.7:  # 70% of meaningful words should come from content
                    validation_result['issues'].append(f"Low word coverage from source content: {word_coverage:.2f}")
                    validation_result['confidence'] *= 0.8
        
        # Update validation status
        if validation_result['confidence'] < 0.6:
            validation_result['is_valid'] = False
        
        return validation_result


class AIDocumentChatbot:
    """Main AI-powered chatbot with strict document-only restrictions"""
    
    def __init__(self, ai_model: AIModelInterface):
        self.ai_model = ai_model
        self.prompt_engine = DocumentOnlyPromptEngine()
        self.content_extractor = DocumentContentExtractor()
        self.document_content = None
        self.searcher = None
        self.validator = None
    
    def load_document(self, pdf_text: str) -> Dict:
        """Load and process a PDF document"""
        
        print("📄 Processing PDF document...")
        self.document_content = self.content_extractor.extract_pdf_content(pdf_text)
        self.searcher = SmartDocumentSearcher(self.document_content)
        self.validator = AIResponseValidator(self.document_content)
        
        return {
            'status': 'success',
            'total_sentences': self.document_content['index_info']['total_sentences'],
            'total_words': self.document_content['index_info']['total_words'],
            'sections_found': len(self.document_content['organized_content'])
        }
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using AI with document-only restrictions"""
        
        if not self.document_content:
            return {
                'answer': '📁 Por favor, sube un documento PDF primero.',
                'confidence': 1.0,
                'source_count': 0,
                'processing_info': 'No document loaded'
            }
        
        # Step 1: Search for relevant content
        print(f"🔍 Searching for content related to: {question}")
        relevant_content = self.searcher.search_all_relevant_content(question)
        
        # Step 2: Determine question type and create appropriate prompt
        question_type = self._determine_question_type(question)
        prompt = self._create_appropriate_prompt(question_type, question, relevant_content)
        
        # Step 3: Get AI response
        print(f"🤖 Generating AI response using {len(relevant_content)} content pieces...")
        ai_response = self.ai_model.generate_response(prompt)
        
        # Step 4: Validate response
        validation_result = self.validator.validate_response(question, ai_response, relevant_content)
        
        # Step 5: Handle validation results
        final_answer = ai_response
        if not validation_result['is_valid']:
            print(f"⚠️ Validation issues: {validation_result['issues']}")
            # If validation fails, fall back to safe response
            if relevant_content:
                final_answer = f"📖 Según el documento: {relevant_content[0][:300]}..."
            else:
                final_answer = "Esta información no se encuentra en el documento."
        
        return {
            'answer': final_answer,
            'confidence': validation_result['confidence'],
            'source_count': len(relevant_content),
            'processing_info': {
                'question_type': question_type,
                'content_found': len(relevant_content),
                'validation_passed': validation_result['is_valid'],
                'validation_issues': validation_result['issues']
            }
        }
    
    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question being asked"""
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ['qué es', 'que es', 'define', 'definir', 'significa']):
            return 'definition'
        elif any(phrase in question_lower for phrase in ['historia', 'cronología', 'timeline', 'desarrollo histórico']):
            return 'historical'
        elif any(phrase in question_lower for phrase in ['cómo', 'como', 'proceso', 'método']):
            return 'process'
        elif any(phrase in question_lower for phrase in ['por qué', 'porque', 'razón', 'motivo']):
            return 'explanation'
        else:
            return 'general'
    
    def _create_appropriate_prompt(self, question_type: str, question: str, relevant_content: List[str]) -> str:
        """Create appropriate prompt based on question type"""
        
        if question_type == 'definition':
            return self.prompt_engine.create_definition_prompt(question, relevant_content)
        elif question_type == 'historical':
            return self.prompt_engine.create_historical_prompt(question, relevant_content)
        else:
            return self.prompt_engine.create_search_prompt(question, relevant_content)
    
    def get_debug_info(self, question: str) -> Dict:
        """Get detailed debug information for troubleshooting"""
        
        if not self.document_content:
            return {'error': 'No document loaded'}
        
        relevant_content = self.searcher.search_all_relevant_content(question)
        question_type = self._determine_question_type(question)
        
        return {
            'question': question,
            'question_type': question_type,
            'relevant_content_count': len(relevant_content),
            'relevant_content': relevant_content[:3],  # Show first 3
            'document_stats': self.document_content['index_info'],
            'sections_available': list(self.document_content['organized_content'].keys())
        }


# Convenience functions for easy setup
def create_openai_chatbot(api_key: str, model: str = "gpt-3.5-turbo") -> AIDocumentChatbot:
    """Create chatbot with OpenAI model"""
    ai_model = OpenAIModel(api_key, model)
    return AIDocumentChatbot(ai_model)


def create_claude_chatbot(api_key: str, model: str = "claude-3-sonnet-20240229") -> AIDocumentChatbot:
    """Create chatbot with Claude model"""
    ai_model = ClaudeModel(api_key, model)
    return AIDocumentChatbot(ai_model)


def create_gemini_chatbot(api_key: str, model: str = "gemini-pro") -> AIDocumentChatbot:
    """Create chatbot with Gemini model"""
    ai_model = GeminiModel(api_key, model)
    return AIDocumentChatbot(ai_model)


# Mock AI model for testing without API keys
class MockAIModel(AIModelInterface):
    """Mock AI model for testing without real API keys"""
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        # Simple mock response based on prompt content
        if "no relevant content" in prompt or "[No relevant content found]" in prompt:
            return "Esta información no se encuentra en el documento."
        
        # Extract content from prompt
        content_sections = re.findall(r'\[CONTENT \d+\]: (.+?)(?=\[CONTENT \d+\]|INSTRUCTIONS:)', prompt, re.DOTALL)
        
        if content_sections:
            # Combine and return a simple response
            combined = " ".join(content_sections[:2])  # Use first 2 content pieces
            return f"📖 Según el documento: {combined[:300]}..."
        
        return "Esta información no se encuentra en el documento."


def create_mock_chatbot() -> AIDocumentChatbot:
    """Create chatbot with mock AI model for testing"""
    ai_model = MockAIModel()
    return AIDocumentChatbot(ai_model)
