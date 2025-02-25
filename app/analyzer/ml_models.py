"""
Machine learning models and text processing utilities for legal document analysis.
"""

import re
import spacy
import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
import threading
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global model instances with thread-safe initialization
_nlp_lock = threading.Lock()
_model_lock = threading.Lock()
_nlp_instance = None
_model_instance = None

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

def get_spacy_model():
    global _nlp_instance
    if _nlp_instance is None:
        with _nlp_lock:
            if _nlp_instance is None:
                # Use the smaller model and disable unnecessary components
                _nlp_instance = spacy.load('en_core_web_sm', 
                                         disable=['ner', 'parser', 'textcat', 'entity_linker', 'entity_ruler', 'sentencizer'])
                if DEVICE.type == 'cuda':
                    # Enable GPU acceleration for spaCy if available
                    _nlp_instance.to(DEVICE)
    return _nlp_instance

def get_transformer_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                # Use a smaller, faster transformer model
                _model_instance = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=DEVICE)
    return _model_instance

class TextProcessor:
    """Text preprocessing and basic NLP operations"""
    
    def __init__(self):
        self._stop_words = set(stopwords.words('english'))
        self._lemmatizer = WordNetLemmatizer()
        self.tfidf = torch.nn.Module()
    
    @property
    def nlp(self):
        return get_spacy_model()
    
    @property
    def model(self):
        return get_transformer_model()
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Cache preprocessed text to avoid redundant processing"""
        # Basic cleaning
        text = text.lower().strip()
        
        # Tokenize and clean in one pass
        tokens = [
            self._lemmatizer.lemmatize(token)
            for token in word_tokenize(text)
            if token.isalnum() and token not in self._stop_words
        ]
        
        return ' '.join(tokens)
    
    @lru_cache(maxsize=100)
    def compute_embeddings(self, text: str) -> np.ndarray:
        """Cache embeddings for frequently analyzed text"""
        with torch.no_grad():  # Disable gradient computation for inference
            embeddings = self.model.encode([text], convert_to_tensor=True)
            return embeddings.cpu().numpy()[0] if DEVICE.type == 'cuda' else embeddings[0]
        
    def compute_similarity(self, doc1: str, doc2: str) -> float:
        """Compute cosine similarity between two documents using GPU if available"""
        # Use cached embeddings
        with torch.no_grad():
            emb1 = torch.tensor(self.compute_embeddings(self.preprocess_text(doc1))).to(DEVICE)
            emb2 = torch.tensor(self.compute_embeddings(self.preprocess_text(doc2))).to(DEVICE)
            
            if DEVICE.type == 'cuda':
                similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                return float(similarity.cpu().numpy()[0])
            else:
                return float(cosine_similarity([emb1.numpy()], [emb2.numpy()])[0][0])
    
    def extract_phrases(self, text: str) -> List[str]:
        """Extract key phrases using SpaCy"""
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun chunks
        phrases.extend([chunk.text for chunk in doc.noun_chunks])
        
        # Extract named entities
        phrases.extend([ent.text for ent in doc.ents])
        
        return list(set(phrases))

class SemanticAnalyzer:
    """Advanced semantic analysis for legal documents"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.nlp = get_spacy_model()
    
    def analyze_semantics(self, text: str) -> Dict:
        """Analyze semantic structure and quality of legal text"""
        doc = self.nlp(text)
        
        # Analyze sentence complexity
        sentence_lengths = [len(sent) for sent in doc.sents]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Analyze vocabulary diversity
        unique_words = set(token.text.lower() for token in doc if token.is_alpha)
        total_words = sum(1 for token in doc if token.is_alpha)
        lexical_diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        # Analyze legal terminology density
        legal_terms = self._get_legal_terms()
        legal_term_count = sum(1 for term in legal_terms if term in text.lower())
        legal_density = legal_term_count / len(doc) if len(doc) > 0 else 0
        
        # Calculate scores
        clarity_score = max(0, 1 - (avg_sentence_length / 50))
        diversity_score = min(lexical_diversity * 2, 1)
        legal_score = min(legal_density * 5, 1)
        quality_score = (clarity_score + diversity_score + legal_score) / 3
        
        return {
            'metrics': {
                'avg_sentence_length': avg_sentence_length,
                'lexical_diversity': lexical_diversity,
                'legal_term_density': legal_density
            },
            'scores': {
                'clarity': clarity_score,
                'diversity': diversity_score,
                'legal_quality': legal_score
            },
            'quality_score': quality_score,
            'key_phrases': self.text_processor.extract_phrases(text)
        }
    
    def _get_legal_terms(self) -> Set[str]:
        """Get common legal terminology"""
        return {
            'agreement', 'contract', 'party', 'parties', 'terms', 'conditions',
            'liability', 'damages', 'termination', 'rights', 'obligations',
            'confidential', 'intellectual property', 'compliance', 'warranty',
            'plaintiff', 'defendant', 'affidavit', 'jurisdiction',
            'whereas', 'hereinafter', 'pursuant', 'notwithstanding', 'forthwith',
            'indemnify', 'arbitration', 'consideration', 'force majeure',
            'governing law', 'severability', 'assignment', 'modification'
        }

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract important phrases using noun chunks and named entities"""
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Only phrases with 2+ words
                phrases.append(chunk.text.lower())
                
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'LAW', 'GPE', 'DATE', 'MONEY']:
                phrases.append(ent.text.lower())
                
        return list(set(phrases))  # Remove duplicates
        
    def analyze_legal_context(self, text: str) -> Dict:
        """Analyze the legal context and relationships in the text"""
        doc = self.nlp(text)
        
        # Get comprehensive clause analysis
        clauses = []
        clause_dependencies = []
        
        for sent in doc.sents:
            clause = self.legal_matcher.analyze_clause(sent.text)
            if clause:
                clause_dict = {
                    'type': clause.type.value,
                    'strength': clause.strength,
                    'importance_level': clause.importance_level,
                    'risk_level': clause.risk_level,
                    'complexity_score': clause.complexity_score,
                    'key_terms': clause.key_terms,
                    'text': clause.text
                }
                clause_dict['weighted_score'] = self.calculate_weighted_score(clause_dict)
                clauses.append(clause_dict)
                if clause.dependencies:
                    clause_dependencies.extend(clause.dependencies)
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_strength': np.mean([c['strength'] for c in clauses]) if clauses else 0,
            'avg_importance': np.mean([c['importance_level'] for c in clauses]) if clauses else 0,
            'avg_risk': np.mean([c['risk_level'] for c in clauses]) if clauses else 0,
            'avg_complexity': np.mean([c['complexity_score'] for c in clauses]) if clauses else 0,
            'avg_score': np.mean([c['weighted_score'] for c in clauses]) if clauses else 0
        }
        
        return {
            'clauses': clauses,
            'overall_metrics': overall_metrics,
            'clause_dependencies': clause_dependencies,
            'legal_terms': self.legal_matcher.identify_legal_terms(text)
        }
        
    def analyze_clause_hierarchy(self, text: str) -> Dict:
        """Analyze the hierarchical structure of clauses"""
        doc = self.nlp(text)
        
        # Track section levels and their clauses
        hierarchy = {}
        current_level = 0
        current_section = None
        
        for sent in doc.sents:
            # Check for section headers
            if re.match(r'^(?:Section|Article|Clause)\s+\d+', sent.text):
                current_section = sent.text
                current_level += 1
            
            clause = self.legal_matcher.analyze_clause(sent.text)
            if clause:
                if current_section not in hierarchy:
                    hierarchy[current_section or f"Level_{current_level}"] = []
                hierarchy[current_section or f"Level_{current_level}"].append({
                    'type': clause.type.value,
                    'text': clause.text,
                    'strength': clause.strength
                })
                
        return hierarchy
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using TF-IDF and cosine similarity"""
        # Fit and transform the texts
        tfidf_matrix = self.text_processor.tfidf.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        
    def get_semantic_score(self, text: str, reference_texts: List[str]) -> Tuple[float, Dict]:
        """Calculate a comprehensive semantic understanding score"""
        # Get detailed analysis
        structure_analysis = self.analyze_semantic_structure(text)
        legal_context = self.analyze_legal_context(text)
        hierarchy = self.analyze_clause_hierarchy(text)
        
        # Calculate similarity with reference texts
        similarities = []
        for ref_text in reference_texts:
            sim_score = self.calculate_semantic_similarity(text, ref_text)
            similarities.append(sim_score)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Calculate component scores
        clause_quality_score = legal_context['overall_metrics']['avg_score']
        structure_score = min(100, len(structure_analysis['key_phrases']) * 5)
        coverage_score = min(100, len(structure_analysis['clauses']) * 10)
        
        # Risk and complexity adjustments
        risk_adjustment = (1 - legal_context['overall_metrics']['avg_risk']) * 100
        complexity_adjustment = (1 - legal_context['overall_metrics']['avg_complexity']) * 100
        
        # Calculate final score with sophisticated weighting
        weights = {
            'similarity': 0.2,
            'clause_quality': 0.3,
            'structure': 0.15,
            'coverage': 0.15,
            'risk': 0.1,
            'complexity': 0.1
        }
        
        final_score = (
            weights['similarity'] * (avg_similarity * 100) +
            weights['clause_quality'] * clause_quality_score +
            weights['structure'] * structure_score +
            weights['coverage'] * coverage_score +
            weights['risk'] * risk_adjustment +
            weights['complexity'] * complexity_adjustment
        )
        
        details = {
            'similarity_score': round(avg_similarity * 100, 2),
            'clause_quality_score': round(clause_quality_score, 2),
            'structure_score': round(structure_score, 2),
            'coverage_score': round(coverage_score, 2),
            'risk_adjustment': round(risk_adjustment, 2),
            'complexity_adjustment': round(complexity_adjustment, 2),
            'clause_metrics': structure_analysis['clause_metrics'],
            'overall_metrics': legal_context['overall_metrics'],
            'clauses': structure_analysis['clauses'],
            'hierarchy': hierarchy
        }
        
        return final_score, details

    def calculate_weighted_score(self, clause: Dict) -> float:
        """Calculate weighted score for a clause based on multiple factors"""
        weights = {
            'strength': 0.3,
            'importance': 0.3,
            'risk': 0.2,
            'complexity': 0.2
        }
        
        return (
            weights['strength'] * clause['strength'] +
            weights['importance'] * clause['importance_level'] +
            weights['risk'] * (1 - clause['risk_level']) +  # Lower risk is better
            weights['complexity'] * (1 - clause['complexity_score'])  # Lower complexity is better
        ) * 100
        
    def analyze_semantic_structure(self, text: str) -> Dict:
        """Analyze the semantic structure of the contract text"""
        doc = self.nlp(text)
        
        # Analyze clauses with enhanced metrics
        clauses = []
        for sent in doc.sents:
            clause = self.legal_matcher.analyze_clause(sent.text)
            if clause:
                clause_dict = {
                    'type': clause.type.value,
                    'strength': clause.strength,
                    'importance_level': clause.importance_level,
                    'risk_level': clause.risk_level,
                    'complexity_score': clause.complexity_score,
                    'key_terms': clause.key_terms,
                    'dependencies': clause.dependencies,
                    'text': clause.text
                }
                clause_dict['weighted_score'] = self.calculate_weighted_score(clause_dict)
                clauses.append(clause_dict)
        
        # Analyze clause distribution and quality
        clause_metrics = {}
        for clause in clauses:
            metrics = clause_metrics.get(clause['type'], {
                'count': 0,
                'avg_strength': 0,
                'avg_importance': 0,
                'avg_risk': 0,
                'avg_complexity': 0,
                'total_score': 0
            })
            metrics['count'] += 1
            metrics['avg_strength'] += clause['strength']
            metrics['avg_importance'] += clause['importance_level']
            metrics['avg_risk'] += clause['risk_level']
            metrics['avg_complexity'] += clause['complexity_score']
            metrics['total_score'] += clause['weighted_score']
            clause_metrics[clause['type']] = metrics
        
        # Calculate averages
        for metrics in clause_metrics.values():
            count = metrics['count']
            if count > 0:
                metrics['avg_strength'] /= count
                metrics['avg_importance'] /= count
                metrics['avg_risk'] /= count
                metrics['avg_complexity'] /= count
                metrics['avg_score'] = metrics['total_score'] / count
        
        return {
            'clauses': clauses,
            'clause_metrics': clause_metrics,
            'key_phrases': self.extract_key_phrases(text)
        }
