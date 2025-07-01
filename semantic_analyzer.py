"""
Semantic Analysis Module - Core engine for true semantic document understanding

This module implements the required solution approach for PDF chatbot:
1. True semantic search (not template matching)
2. Line-by-line document analysis
3. Semantic similarity-based content extraction
4. Relevance validation before response generation
"""

import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


class SemanticQuestionAnalyzer:
    """Analyzes questions to understand semantic intent and required information type"""
    
    def __init__(self):
        self.question_types = {
            'definition': ['qué es', 'que es', 'what is', 'define', 'definir', 'significa', 'mean'],
            'explanation': ['cómo', 'como', 'how', 'por qué', 'porque', 'why', 'explica', 'explain'],
            'characteristics': ['características', 'properties', 'propiedades', 'features'],
            'examples': ['ejemplo', 'examples', 'por ejemplo', 'for example'],
            'process': ['proceso', 'process', 'pasos', 'steps', 'etapas', 'stages'],
            'comparison': ['diferencia', 'difference', 'comparar', 'compare', 'versus', 'vs'],
            'application': ['aplicación', 'application', 'uso', 'use', 'utilizar', 'aplicar']
        }
    
    def semantic_question_analysis(self, question: str) -> Dict:
        """
        Understand what the user is actually asking
        
        Args:
            question: User's question
            
        Returns:
            Dict with question analysis including type, key concepts, and search strategy
        """
        question_lower = question.lower().strip()
        
        # Extract key concepts (removing stop words)
        stop_words = {
            'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para',
            'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the',
            'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as'
        }
        
        words = question_lower.split()
        key_concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Determine question type
        question_type = 'general'
        for q_type, patterns in self.question_types.items():
            for pattern in patterns:
                if pattern in question_lower:
                    question_type = q_type
                    break
            if question_type != 'general':
                break
        
        # Extract specific entities (proper nouns, technical terms)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        
        # Determine search priority
        search_priority = self._determine_search_priority(question_type, key_concepts, entities)
        
        return {
            'original_question': question,
            'question_type': question_type,
            'key_concepts': key_concepts,
            'entities': entities,
            'search_priority': search_priority,
            'requires_exact_match': len(entities) > 0 or question_type == 'definition'
        }
    
    def _determine_search_priority(self, question_type: str, key_concepts: List[str], entities: List[str]) -> str:
        """Determine search strategy priority"""
        if entities:
            return 'exact_entities'
        elif question_type == 'definition':
            return 'definition_patterns'
        elif len(key_concepts) <= 2:
            return 'precise_semantic'
        else:
            return 'broad_semantic'


class LineByLineDocumentAnalyzer:
    """Performs exhaustive line-by-line analysis of document content"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        try:
            self.model = SentenceTransformer(model_name)
        except:
            # Fallback models
            fallback_models = [
                'distiluse-base-multilingual-cased',
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2'
            ]
            for fallback in fallback_models:
                try:
                    self.model = SentenceTransformer(fallback)
                    break
                except:
                    continue
    
    def line_by_line_document_search(self, question_analysis: Dict, document_lines: List[str]) -> Tuple[List[str], List[float], List[str]]:
        """
        Search through every single line of the document systematically
        
        Args:
            question_analysis: Analysis from SemanticQuestionAnalyzer
            document_lines: All lines from the document
            
        Returns:
            Tuple of (relevant_chunks, similarity_scores, specific_lines)
        """
        if not document_lines or not question_analysis:
            return [], [], []
        
        question = question_analysis['original_question']
        key_concepts = question_analysis['key_concepts']
        question_type = question_analysis['question_type']
        
        # Generate question embedding
        question_embedding = self.model.encode([question])[0]
        
        # Analyze each line
        line_results = []
        
        for i, line in enumerate(document_lines):
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue
            
            # Skip obvious metadata/headers
            if self._is_metadata_line(line):
                continue
            
            # Calculate semantic similarity
            line_embedding = self.model.encode([line])[0]
            semantic_score = cosine_similarity([question_embedding], [line_embedding])[0][0]
            
            # Calculate concept overlap score
            concept_score = self._calculate_concept_overlap(key_concepts, line)
            
            # Calculate pattern matching score for specific question types
            pattern_score = self._calculate_pattern_score(question_type, question, line)
            
            # Combined relevance score
            combined_score = (semantic_score * 0.4 + concept_score * 0.3 + pattern_score * 0.3)
            
            # Check if line meets minimum relevance threshold
            if combined_score >= 0.35 or semantic_score >= 0.5:
                line_results.append({
                    'line': line,
                    'line_index': i,
                    'semantic_score': semantic_score,
                    'concept_score': concept_score,
                    'pattern_score': pattern_score,
                    'combined_score': combined_score
                })
        
        # Sort by relevance
        line_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Extract top results
        top_results = line_results[:5]  # Limit to top 5 most relevant
        
        if top_results:
            relevant_chunks = [result['line'] for result in top_results]
            scores = [result['combined_score'] for result in top_results]
            specific_lines = relevant_chunks.copy()  # Same as chunks in line-by-line analysis
            
            return relevant_chunks, scores, specific_lines
        
        return [], [], []
    
    def _is_metadata_line(self, line: str) -> bool:
        """Check if line is metadata/header that should be skipped"""
        line_lower = line.lower().strip()
        
        metadata_patterns = [
            r'^\d+$',  # Just numbers
            r'^página \d+',  # Page numbers
            r'^page \d+',
            r'isbn.*\d',
            r'^capítulo \d+',
            r'^chapter \d+',
            r'^tabla de contenido',
            r'^table of contents',
            r'^índice',
            r'^bibliografía',
            r'^referencias'
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, line_lower):
                return True
        
        # Check if line is mostly uppercase (likely header)
        if len(line) < 100 and line.isupper():
            return True
        
        return False
    
    def _calculate_concept_overlap(self, key_concepts: List[str], line: str) -> float:
        """Calculate overlap between question concepts and line content"""
        if not key_concepts:
            return 0.0
        
        line_words = set(line.lower().split())
        concept_matches = sum(1 for concept in key_concepts if concept.lower() in line_words)
        
        return concept_matches / len(key_concepts)
    
    def _calculate_pattern_score(self, question_type: str, question: str, line: str) -> float:
        """Calculate pattern-specific scoring based on question type"""
        line_lower = line.lower()
        question_lower = question.lower()
        
        # Definition patterns
        if question_type == 'definition':
            definition_indicators = ['es', 'son', 'se define como', 'significa', 'consiste en', ':']
            if any(indicator in line_lower for indicator in definition_indicators):
                return 0.8
        
        # Explanation patterns
        elif question_type == 'explanation':
            explanation_indicators = ['porque', 'debido a', 'ya que', 'permite', 'ayuda a']
            if any(indicator in line_lower for indicator in explanation_indicators):
                return 0.7
        
        # Example patterns
        elif question_type == 'examples':
            example_indicators = ['por ejemplo', 'como', 'ejemplo', 'casos']
            if any(indicator in line_lower for indicator in example_indicators):
                return 0.7
        
        # Look for exact entity matches
        entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', question)
        for entity in entities:
            if entity.lower() in line_lower:
                return 0.9
        
        return 0.0


class SemanticRelevanceValidator:
    """Validates semantic match between question and content before returning response"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def validate_answer_relevance(self, question: str, answer: str, confidence_threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Ensure response actually relates to the question
        
        Args:
            question: Original question
            answer: Proposed answer
            confidence_threshold: Minimum confidence required
            
        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        if not question or not answer:
            return False, 0.0
        
        # Generate embeddings
        question_embedding = self.model.encode([question])[0]
        answer_embedding = self.model.encode([answer])[0]
        
        # Calculate semantic similarity
        semantic_similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
        
        # Calculate keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove stop words
        stop_words = {
            'que', 'es', 'son', 'la', 'el', 'de', 'en', 'y', 'a', 'un', 'una', 'para',
            'con', 'por', 'se', 'del', 'las', 'los', 'como', 'what', 'is', 'are', 'the',
            'of', 'in', 'and', 'to', 'a', 'an', 'for', 'with', 'by', 'from', 'as'
        }
        
        meaningful_question_words = question_words - stop_words
        meaningful_answer_words = answer_words - stop_words
        
        if meaningful_question_words:
            keyword_overlap = len(meaningful_question_words.intersection(meaningful_answer_words)) / len(meaningful_question_words)
        else:
            keyword_overlap = 0.0
        
        # Combined confidence score
        confidence_score = (semantic_similarity * 0.7 + keyword_overlap * 0.3)
        
        # Validation logic
        is_relevant = (
            confidence_score >= confidence_threshold and
            semantic_similarity >= 0.4 and
            keyword_overlap >= 0.1
        )
        
        return is_relevant, confidence_score


class SemanticAnswerExtractor:
    """Extracts relevant answers from document content"""
    
    def extract_relevant_answer(self, question_analysis: Dict, matching_lines: List[str]) -> str:
        """
        Get the actual answer from matching lines
        
        Args:
            question_analysis: Analysis from SemanticQuestionAnalyzer
            matching_lines: Lines that matched the question
            
        Returns:
            Extracted answer text
        """
        if not matching_lines:
            return ""
        
        question_type = question_analysis['question_type']
        
        # For definitions, look for the most complete definition
        if question_type == 'definition':
            return self._extract_definition(matching_lines)
        
        # For explanations, combine relevant explanation parts
        elif question_type == 'explanation':
            return self._extract_explanation(matching_lines)
        
        # For other types, return the most relevant line(s)
        else:
            return self._extract_general_answer(matching_lines)
    
    def _extract_definition(self, lines: List[str]) -> str:
        """Extract definition-style answers"""
        # Look for lines with definition patterns
        definition_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in ['es', 'son', ':', 'se define', 'significa']):
                definition_lines.append(line)
        
        if definition_lines:
            # Return the most complete definition (longest with definition indicators)
            return max(definition_lines, key=len)
        
        # Return first line if no clear definition pattern
        return lines[0] if lines else ""
    
    def _extract_explanation(self, lines: List[str]) -> str:
        """Extract explanation-style answers"""
        # Combine multiple relevant lines for comprehensive explanation
        if len(lines) == 1:
            return lines[0]
        elif len(lines) <= 3:
            return " ".join(lines)
        else:
            # Return top 2 most relevant lines
            return " ".join(lines[:2])
    
    def _extract_general_answer(self, lines: List[str]) -> str:
        """Extract general answers"""
        if len(lines) == 1:
            return lines[0]
        else:
            # Return the longest, most comprehensive line
            return max(lines, key=len)


class SemanticResponseFormatter:
    """Formats final responses with proper context analysis"""
    
    def format_final_response(self, answer: str, confidence_level: float) -> str:
        """
        Format output with confidence indicators
        
        Args:
            answer: Extracted answer
            confidence_level: Confidence score
            
        Returns:
            Formatted response
        """
        if not answer:
            return "Esta información no se encuentra en el documento."
        
        # Remove document artifacts
        cleaned_answer = self._clean_answer(answer)
        
        if confidence_level >= 0.8:
            return f"📖 {cleaned_answer}"
        elif confidence_level >= 0.6:
            return f"📄 Según el documento: {cleaned_answer}"
        elif confidence_level >= 0.4:
            return f"💭 El documento menciona: {cleaned_answer}"
        else:
            return "Esta información no se encuentra en el documento."
    
    def _clean_answer(self, answer: str) -> str:
        """Clean extracted answer from document artifacts"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', answer.strip())
        
        # Remove page numbers and citations
        cleaned = re.sub(r'\b\d+\s*$', '', cleaned)  # Remove trailing numbers
        cleaned = re.sub(r'^\d+\s*', '', cleaned)   # Remove leading numbers
        
        return cleaned.strip()


# Main integration class
class TrueSemanticPDFAnalyzer:
    """
    Main class that integrates all semantic analysis components
    Implements the complete solution approach as specified
    """
    
    def __init__(self):
        self.question_analyzer = SemanticQuestionAnalyzer()
        self.document_analyzer = LineByLineDocumentAnalyzer()
        self.relevance_validator = SemanticRelevanceValidator(self.document_analyzer.model)
        self.answer_extractor = SemanticAnswerExtractor()
        self.response_formatter = SemanticResponseFormatter()
    
    def process_document_query(self, question: str, document_text: str) -> str:
        """
        Complete semantic processing pipeline
        
        Args:
            question: User's question
            document_text: Full document text
            
        Returns:
            Final response
        """
        # Step 1: Analyze question semantically
        question_analysis = self.question_analyzer.semantic_question_analysis(question)
        
        # Step 2: Convert document to lines for analysis
        document_lines = self._prepare_document_lines(document_text)
        
        # Step 3: Line-by-line search
        relevant_chunks, scores, specific_lines = self.document_analyzer.line_by_line_document_search(
            question_analysis, document_lines
        )
        
        # Step 4: Extract answer if relevant content found
        if relevant_chunks:
            extracted_answer = self.answer_extractor.extract_relevant_answer(
                question_analysis, specific_lines
            )
            
            # Step 5: Validate relevance
            if extracted_answer:
                is_relevant, confidence = self.relevance_validator.validate_answer_relevance(
                    question, extracted_answer
                )
                
                if is_relevant:
                    # Step 6: Format final response
                    return self.response_formatter.format_final_response(extracted_answer, confidence)
        
        # If no relevant information found
        return "Esta información no se encuentra en el documento."
    
    def _prepare_document_lines(self, document_text: str) -> List[str]:
        """Prepare document text for line-by-line analysis"""
        # Split by sentences and paragraphs
        lines = []
        
        # Split by paragraphs first
        paragraphs = document_text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Split long paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 20:  # Only include substantial sentences
                    lines.append(sentence)
        
        return lines
