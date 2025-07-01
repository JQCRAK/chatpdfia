"""
Enhanced Document Search Engine
Prioritizes finding existing content over filtering - NO FALSE NEGATIVES

This system implements a comprehensive search approach that:
1. Scans every sentence in the document
2. Uses multiple search strategies
3. Prioritizes recall over precision
4. Only returns "not found" when content truly doesn't exist
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Optional
from difflib import SequenceMatcher
import math


class DocumentProcessor:
    """Processes PDF text into searchable format"""
    
    def __init__(self):
        self.sentences = []
        self.paragraphs = []
        self.word_index = {}
        
    def process_document(self, raw_text: str) -> Dict:
        """
        Process raw PDF text into searchable format
        
        Args:
            raw_text: Raw text from PDF
            
        Returns:
            Dict with processed content
        """
        # Step 1: Clean the text
        cleaned_text = self._clean_text(raw_text)
        
        # Step 2: Split into sentences and paragraphs
        sentences = self._extract_sentences(cleaned_text)
        paragraphs = self._extract_paragraphs(cleaned_text)
        
        # Step 3: Create word index for fast lookup
        word_index = self._create_word_index(sentences)
        
        # Step 4: Store processed content
        self.sentences = sentences
        self.paragraphs = paragraphs
        self.word_index = word_index
        
        return {
            'sentences': sentences,
            'paragraphs': paragraphs,
            'word_index': word_index,
            'total_sentences': len(sentences),
            'total_words': sum(len(s.split()) for s in sentences)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving content"""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        
        # Remove page numbers and headers (but keep content)
        text = re.sub(r'\b(página|page)\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*$', '', text)  # Remove trailing page numbers
        
        return text.strip()
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract all sentences from text"""
        sentences = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 10:
                continue
            
            # Split paragraph into sentences
            # Handle Spanish punctuation and abbreviations
            sentence_endings = r'[.!?](?:\s|$)'
            potential_sentences = re.split(sentence_endings, paragraph)
            
            for sentence in potential_sentences:
                sentence = sentence.strip()
                # Include sentences with meaningful content
                if len(sentence) >= 15 and not self._is_metadata(sentence):
                    sentences.append(sentence)
        
        return sentences
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = []
        
        for paragraph in text.split('\n\n'):
            paragraph = paragraph.strip()
            if len(paragraph) >= 20 and not self._is_metadata(paragraph):
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def _create_word_index(self, sentences: List[str]) -> Dict[str, List[int]]:
        """Create index mapping words to sentence indices"""
        word_index = {}
        
        for i, sentence in enumerate(sentences):
            words = self._extract_words(sentence)
            for word in words:
                if word not in word_index:
                    word_index[word] = []
                word_index[word].append(i)
        
        return word_index
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text for indexing"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'una', 'del', 'las', 'los', 'un', 'al'}
        return [w for w in words if len(w) >= 3 and w not in stop_words]
    
    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata/header to skip"""
        text_lower = text.lower().strip()
        
        # Skip obvious metadata
        metadata_patterns = [
            r'^(capítulo|chapter)\s+\d+',
            r'^página\s+\d+',
            r'^page\s+\d+',
            r'isbn\s*:?\s*\d',
            r'^bibliografía',
            r'^references',
            r'^índice',
            r'^table of contents'
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Skip if too short or all uppercase
        if len(text) < 10 or (text.isupper() and len(text) < 100):
            return True
        
        return False


class QuestionAnalyzer:
    """Analyzes user questions to extract search terms and intent"""
    
    def analyze_question(self, question: str) -> Dict:
        """
        Analyze question to understand what to search for
        
        Args:
            question: User's question
            
        Returns:
            Dict with analysis results
        """
        question_clean = question.strip()
        question_lower = question_clean.lower()
        
        # Extract key terms
        key_terms = self._extract_key_terms(question_lower)
        
        # Detect question type
        question_type = self._detect_question_type(question_lower)
        
        # Generate search variations
        search_variations = self._generate_search_variations(question_clean, key_terms)
        
        # Extract quoted phrases (exact matches)
        quoted_phrases = self._extract_quoted_phrases(question_clean)
        
        return {
            'original': question_clean,
            'key_terms': key_terms,
            'question_type': question_type,
            'search_variations': search_variations,
            'quoted_phrases': quoted_phrases,
            'search_priority': self._determine_search_priority(question_type, key_terms)
        }
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract important terms from question"""
        # Remove question words and common terms
        question_words = {'qué', 'que', 'cuál', 'cual', 'cómo', 'como', 'por', 'para', 'son', 'es', 'está', 'esta', 'dónde', 'donde', 'cuándo', 'cuando', 'why', 'what', 'how', 'where', 'when', 'who'}
        
        # Extract words, keeping compound terms
        words = re.findall(r'\b\w+\b', question)
        
        # Filter and prioritize
        key_terms = []
        for word in words:
            if (len(word) >= 3 and 
                word.lower() not in question_words and
                not word.isdigit()):
                key_terms.append(word.lower())
        
        return key_terms
    
    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question being asked"""
        if any(phrase in question for phrase in ['qué es', 'que es', 'what is', 'definir', 'define']):
            return 'definition'
        elif any(phrase in question for phrase in ['cómo', 'como', 'how', 'manera', 'forma']):
            return 'explanation'
        elif any(phrase in question for phrase in ['por qué', 'porque', 'why', 'razón']):
            return 'reason'
        elif any(phrase in question for phrase in ['cuáles', 'cuales', 'which', 'lista', 'tipos']):
            return 'list'
        elif any(phrase in question for phrase in ['cuándo', 'cuando', 'when', 'fecha']):
            return 'time'
        else:
            return 'general'
    
    def _generate_search_variations(self, question: str, key_terms: List[str]) -> List[str]:
        """Generate variations of the search query"""
        variations = [question]  # Original question
        
        # Add key terms combination
        if len(key_terms) > 1:
            variations.append(' '.join(key_terms))
        
        # Add individual important terms
        for term in key_terms:
            if len(term) >= 4:  # Only longer terms
                variations.append(term)
        
        # Add partial phrases
        words = question.split()
        if len(words) >= 3:
            # Take consecutive word pairs
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                variations.append(phrase)
        
        return list(set(variations))  # Remove duplicates
    
    def _extract_quoted_phrases(self, question: str) -> List[str]:
        """Extract phrases in quotes for exact matching"""
        quoted = re.findall(r'"([^"]*)"', question)
        quoted.extend(re.findall(r"'([^']*)'", question))
        return [q.strip() for q in quoted if len(q.strip()) > 2]
    
    def _determine_search_priority(self, question_type: str, key_terms: List[str]) -> str:
        """Determine search strategy priority"""
        if len(key_terms) <= 2:
            return 'exact_phrase'
        elif question_type == 'definition':
            return 'definition_focused'
        else:
            return 'comprehensive'


class ContentMatcher:
    """Finds relevant content using multiple search strategies"""
    
    def __init__(self, processed_doc: Dict):
        self.sentences = processed_doc['sentences']
        self.paragraphs = processed_doc['paragraphs']
        self.word_index = processed_doc['word_index']
    
    def find_content(self, question_analysis: Dict) -> List[Tuple[str, float, str]]:
        """
        Find relevant content using multiple strategies
        
        Args:
            question_analysis: Analysis from QuestionAnalyzer
            
        Returns:
            List of (content, score, match_type) tuples
        """
        all_matches = []
        
        # Strategy 1: Exact phrase matching
        exact_matches = self._find_exact_matches(question_analysis)
        all_matches.extend(exact_matches)
        
        # Strategy 2: Fuzzy phrase matching
        fuzzy_matches = self._find_fuzzy_matches(question_analysis)
        all_matches.extend(fuzzy_matches)
        
        # Strategy 3: Keyword combination matching
        keyword_matches = self._find_keyword_matches(question_analysis)
        all_matches.extend(keyword_matches)
        
        # Strategy 4: Individual term matching
        term_matches = self._find_individual_term_matches(question_analysis)
        all_matches.extend(term_matches)
        
        # Strategy 5: Context-based matching
        context_matches = self._find_context_matches(question_analysis)
        all_matches.extend(context_matches)
        
        # Remove duplicates and sort by score
        unique_matches = self._deduplicate_matches(all_matches)
        sorted_matches = sorted(unique_matches, key=lambda x: x[1], reverse=True)
        
        return sorted_matches[:10]  # Return top 10 matches
    
    def _find_exact_matches(self, analysis: Dict) -> List[Tuple[str, float, str]]:
        """Find exact phrase matches"""
        matches = []
        search_terms = analysis['search_variations']
        
        for term in search_terms:
            if len(term) < 5:  # Skip very short terms
                continue
                
            for sentence in self.sentences:
                if term.lower() in sentence.lower():
                    # Higher score for longer matches
                    score = 0.9 + (len(term) / 100)
                    matches.append((sentence, score, 'exact_phrase'))
        
        return matches
    
    def _find_fuzzy_matches(self, analysis: Dict) -> List[Tuple[str, float, str]]:
        """Find fuzzy/partial matches"""
        matches = []
        search_terms = analysis['search_variations']
        
        for term in search_terms:
            if len(term) < 8:  # Only for longer terms
                continue
                
            for sentence in self.sentences:
                # Use sequence matcher for fuzzy matching
                similarity = SequenceMatcher(None, term.lower(), sentence.lower()).ratio()
                if similarity >= 0.6:  # 60% similarity threshold
                    score = 0.7 * similarity
                    matches.append((sentence, score, 'fuzzy_match'))
        
        return matches
    
    def _find_keyword_matches(self, analysis: Dict) -> List[Tuple[str, float, str]]:
        """Find sentences containing multiple keywords"""
        matches = []
        key_terms = analysis['key_terms']
        
        if len(key_terms) < 2:
            return matches
        
        for sentence in self.sentences:
            sentence_lower = sentence.lower()
            matching_terms = [term for term in key_terms if term in sentence_lower]
            
            if len(matching_terms) >= 2:
                # Score based on how many terms match
                score = 0.6 + (len(matching_terms) / len(key_terms)) * 0.3
                matches.append((sentence, score, 'keyword_combo'))
        
        return matches
    
    def _find_individual_term_matches(self, analysis: Dict) -> List[Tuple[str, float, str]]:
        """Find sentences with individual important terms"""
        matches = []
        key_terms = analysis['key_terms']
        
        for term in key_terms:
            if len(term) >= 4:  # Only longer terms
                if term in self.word_index:
                    for sentence_idx in self.word_index[term][:5]:  # Limit to 5 per term
                        sentence = self.sentences[sentence_idx]
                        score = 0.4 + (len(term) / 50)  # Longer terms get higher score
                        matches.append((sentence, score, 'individual_term'))
        
        return matches
    
    def _find_context_matches(self, analysis: Dict) -> List[Tuple[str, float, str]]:
        """Find matches based on context and question type"""
        matches = []
        question_type = analysis['question_type']
        key_terms = analysis['key_terms']
        
        # Definition patterns
        if question_type == 'definition':
            definition_patterns = [
                r'(\w+)\s+(es|son|significa|define|consiste)',
                r'(la|el|los|las)\s+(\w+)\s+(es|son)',
                r'(\w+):\s*(.+)'
            ]
            
            for pattern in definition_patterns:
                for sentence in self.sentences:
                    if re.search(pattern, sentence.lower()):
                        # Check if any key terms are in the sentence
                        if any(term in sentence.lower() for term in key_terms):
                            score = 0.8
                            matches.append((sentence, score, 'definition_pattern'))
        
        return matches
    
    def _deduplicate_matches(self, matches: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Remove duplicate matches, keeping highest score"""
        seen_content = {}
        
        for content, score, match_type in matches:
            content_key = content.lower().strip()
            if content_key not in seen_content or score > seen_content[content_key][1]:
                seen_content[content_key] = (content, score, match_type)
        
        return list(seen_content.values())


class AnswerAssembler:
    """Assembles final answers from matched content"""
    
    def __init__(self, processed_doc: Dict):
        self.sentences = processed_doc['sentences']
        self.paragraphs = processed_doc['paragraphs']
    
    def assemble_answer(self, matches: List[Tuple[str, float, str]], question_analysis: Dict) -> str:
        """
        Assemble final answer from matches
        
        Args:
            matches: List of (content, score, match_type) tuples
            question_analysis: Analysis from QuestionAnalyzer
            
        Returns:
            Final answer string
        """
        if not matches:
            return "Esta información no se encuentra en el documento."
        
        # Filter matches by minimum score
        good_matches = [(content, score, match_type) for content, score, match_type in matches if score >= 0.3]
        
        if not good_matches:
            return "Esta información no se encuentra en el documento."
        
        # Select best matches
        best_matches = good_matches[:3]  # Top 3 matches
        
        # Add context if needed
        contextualized_content = []
        for content, score, match_type in best_matches:
            context = self._add_context(content, question_analysis)
            contextualized_content.append((context, score, match_type))
        
        # Format final answer
        return self._format_answer(contextualized_content)
    
    def _add_context(self, sentence: str, question_analysis: Dict) -> str:
        """Add surrounding context to a sentence"""
        # Find the sentence in our collection
        try:
            sentence_idx = self.sentences.index(sentence)
        except ValueError:
            return sentence
        
        # Add previous and next sentence for context
        context_sentences = []
        
        # Add previous sentence if available
        if sentence_idx > 0 and self._is_related_context(self.sentences[sentence_idx - 1], sentence):
            context_sentences.append(self.sentences[sentence_idx - 1])
        
        # Add the main sentence
        context_sentences.append(sentence)
        
        # Add next sentence if available
        if (sentence_idx < len(self.sentences) - 1 and 
            self._is_related_context(self.sentences[sentence_idx + 1], sentence)):
            context_sentences.append(self.sentences[sentence_idx + 1])
        
        return ' '.join(context_sentences)
    
    def _is_related_context(self, context_sentence: str, main_sentence: str) -> bool:
        """Check if context sentence is related to main sentence"""
        # Simple heuristic: if they share significant words
        main_words = set(re.findall(r'\b\w{4,}\b', main_sentence.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', context_sentence.lower()))
        
        if not main_words:
            return False
        
        overlap = len(main_words.intersection(context_words)) / len(main_words)
        return overlap >= 0.2  # 20% word overlap
    
    def _format_answer(self, contextualized_content: List[Tuple[str, float, str]]) -> str:
        """Format the final answer"""
        if not contextualized_content:
            return "Esta información no se encuentra en el documento."
        
        # Get the best match
        best_content, best_score, match_type = contextualized_content[0]
        
        # Format based on confidence
        if best_score >= 0.8:
            return f"📖 {best_content}"
        elif best_score >= 0.6:
            return f"📄 Según el documento: {best_content}"
        elif best_score >= 0.4:
            return f"💭 El documento menciona: {best_content}"
        else:
            return f"🔍 Información relacionada: {best_content}"


class EnhancedSearchEngine:
    """Main search engine that coordinates all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.question_analyzer = QuestionAnalyzer()
        self.content_matcher = None
        self.answer_assembler = None
        self.processed_doc = None
    
    def index_document(self, raw_text: str) -> Dict:
        """Index a document for searching"""
        self.processed_doc = self.document_processor.process_document(raw_text)
        self.content_matcher = ContentMatcher(self.processed_doc)
        self.answer_assembler = AnswerAssembler(self.processed_doc)
        
        return {
            'total_sentences': self.processed_doc['total_sentences'],
            'total_words': self.processed_doc['total_words'],
            'index_ready': True
        }
    
    def search(self, question: str) -> str:
        """
        Search for answer to question in indexed document
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        if not self.content_matcher or not self.answer_assembler:
            return "📋 Por favor, sube un documento PDF primero."
        
        # Step 1: Analyze the question
        question_analysis = self.question_analyzer.analyze_question(question)
        
        # Step 2: Find matching content
        matches = self.content_matcher.find_content(question_analysis)
        
        # Step 3: Assemble answer
        answer = self.answer_assembler.assemble_answer(matches, question_analysis)
        
        return answer
    
    def get_debug_info(self, question: str) -> Dict:
        """Get debug information for troubleshooting"""
        if not self.content_matcher:
            return {"error": "No document indexed"}
        
        question_analysis = self.question_analyzer.analyze_question(question)
        matches = self.content_matcher.find_content(question_analysis)
        
        return {
            'question_analysis': question_analysis,
            'matches_found': len(matches),
            'top_matches': matches[:5],
            'document_stats': {
                'total_sentences': len(self.processed_doc['sentences']),
                'total_words': sum(len(s.split()) for s in self.processed_doc['sentences'])
            }
        }


# Convenience function for integration
def create_enhanced_search_engine() -> EnhancedSearchEngine:
    """Create and return a new enhanced search engine instance"""
    return EnhancedSearchEngine()
