import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import Dict, List, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from .ml_models import SemanticAnalyzer

class ContractScorer:
    def __init__(self):
        """Initialize the contract scorer"""
        self.weights = {
            'clause_coverage': 0.3,
            'risk_score': 0.3,
            'compliance': 0.2,
            'semantic_quality': 0.2
        }
        self.nlp = spacy.load('en_core_web_sm')
        self.semantic_analyzer = SemanticAnalyzer()
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.legal_bert_tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.legal_bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.initialize_fuzzy_system()
        
    def initialize_fuzzy_system(self):
        """Initialize advanced fuzzy logic control system"""
        # Create fuzzy variables
        clarity = ctrl.Antecedent(np.arange(0, 101, 1), 'clarity')
        completeness = ctrl.Antecedent(np.arange(0, 101, 1), 'completeness')
        context = ctrl.Antecedent(np.arange(0, 101, 1), 'context')
        structure = ctrl.Antecedent(np.arange(0, 101, 1), 'structure')
        legal_quality = ctrl.Consequent(np.arange(0, 101, 1), 'legal_quality')

        # Define membership functions
        clarity.automf(5, names=['very_poor', 'poor', 'average', 'good', 'excellent'])
        completeness.automf(5, names=['very_poor', 'poor', 'average', 'good', 'excellent'])
        context.automf(5, names=['very_poor', 'poor', 'average', 'good', 'excellent'])
        structure.automf(5, names=['very_poor', 'poor', 'average', 'good', 'excellent'])
        legal_quality.automf(5, names=['very_poor', 'poor', 'average', 'good', 'excellent'])

        # Define comprehensive rule set
        rules = [
            # Perfect conditions
            ctrl.Rule(
                clarity['excellent'] & completeness['excellent'] & 
                context['excellent'] & structure['excellent'],
                legal_quality['excellent']
            ),
            
            # Good conditions
            ctrl.Rule(
                (clarity['good'] | clarity['excellent']) & 
                (completeness['good'] | completeness['excellent']) &
                (context['good'] | context['excellent']) &
                (structure['good'] | structure['excellent']),
                legal_quality['good']
            ),
            
            # Average conditions with mixed quality
            ctrl.Rule(
                (clarity['average'] & completeness['average']) |
                (context['average'] & structure['average']),
                legal_quality['average']
            ),
            
            # Poor conditions
            ctrl.Rule(
                clarity['poor'] | completeness['poor'] |
                context['poor'] | structure['poor'],
                legal_quality['poor']
            ),
            
            # Critical failures
            ctrl.Rule(
                clarity['very_poor'] | completeness['very_poor'] |
                context['very_poor'] | structure['very_poor'],
                legal_quality['very_poor']
            ),
            
            # Special cases
            ctrl.Rule(
                (clarity['excellent'] & completeness['good'] & 
                 context['average'] & structure['good']) |
                (clarity['good'] & completeness['excellent'] & 
                 context['good'] & structure['average']),
                legal_quality['good']
            ),
            
            # Compensatory rules
            ctrl.Rule(
                (clarity['poor'] & completeness['excellent'] & 
                 context['excellent'] & structure['excellent']) |
                (clarity['excellent'] & completeness['poor'] & 
                 context['excellent'] & structure['excellent']),
                legal_quality['average']
            )
        ]

        # Create control system
        self.quality_ctrl = ctrl.ControlSystem(rules)
        self.quality_sim = ctrl.ControlSystemSimulation(self.quality_ctrl)

    def get_bert_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for text using Legal-BERT"""
        inputs = self.legal_bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.legal_bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Use mean pooling
        
    def semantic_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        embedding1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return float(similarity[0][0])

    def analyze_cross_references(self, text: str) -> Dict:
        """Analyze cross-references and dependencies between contract sections"""
        doc = self.nlp(text)
        references = []
        reference_patterns = ['as described in', 'pursuant to', 'subject to', 'in accordance with', 'as set forth in']
        
        for sent in doc.sents:
            for pattern in reference_patterns:
                if pattern in sent.text.lower():
                    references.append({
                        'sentence': sent.text,
                        'pattern': pattern
                    })
        
        return {
            'references': references,
            'count': len(references)
        }

    def rule_based_score(self, text: str, parameter: Dict) -> float:
        """
        Calculate rule-based score using keyword matching and pattern analysis
        """
        doc = self.nlp(text.lower())
        score = 0
        max_score = len(parameter['keywords'])
        
        # Basic keyword matching
        for keyword in parameter['keywords']:
            if keyword in doc.text:
                score += 1
                
        # Pattern matching for specific structures
        sentences = [sent.text.lower() for sent in doc.sents]
        for keyword in parameter['keywords']:
            for sentence in sentences:
                if keyword in sentence:
                    # Additional points for keywords in proper context
                    if any(indicator in sentence for indicator in ['shall', 'must', 'will', 'agrees to']):
                        score += 0.5
                    
        return min(100, (score / max_score) * 100)

    def nlp_based_score(self, text: str, parameter: Dict) -> Tuple[float, Dict]:
        """Calculate score using advanced NLP and ML techniques"""
        # Get semantic analysis score
        reference_texts = parameter.get('reference_texts', []) + parameter['keywords']
        semantic_score, semantic_details = self.semantic_analyzer.get_semantic_score(text, reference_texts)
        
        # Get legal context analysis
        legal_context = self.semantic_analyzer.analyze_legal_context(text)
        
        # Calculate context score based on legal patterns
        context_score = sum(len(matches) for matches in legal_context['context_analysis'].values())
        context_score = min(100, context_score * 10)  # 10 points per context match
        
        # Calculate structure score based on clause dependencies
        structure_score = len(legal_context['clause_dependencies']) * 20  # 20 points per dependency
        structure_score = min(100, structure_score)
        
        # Combine scores with weights
        weights = {
            'semantic': 0.4,
            'context': 0.3,
            'structure': 0.3
        }
        
        final_score = (
            weights['semantic'] * semantic_score +
            weights['context'] * context_score +
            weights['structure'] * structure_score
        )
        
        details = {
            'semantic_score': round(semantic_score, 2),
            'context_score': round(context_score, 2),
            'structure_score': round(structure_score, 2),
            'semantic_details': semantic_details,
            'legal_context': legal_context
        }
        
        return final_score, details

    def fuzzy_logic_score(self, clarity_score: float, completeness_score: float, 
                         context_score: float, structure_score: float) -> Tuple[float, Dict]:
        """
        Apply enhanced fuzzy logic to evaluate contract quality
        """
        try:
            self.quality_sim.input['clarity'] = clarity_score
            self.quality_sim.input['completeness'] = completeness_score
            self.quality_sim.input['context'] = context_score
            self.quality_sim.input['structure'] = structure_score
            
            self.quality_sim.compute()
            
            quality_score = self.quality_sim.output['legal_quality']
            
            # Calculate confidence based on input spread
            scores = [clarity_score, completeness_score, context_score, structure_score]
            score_std = np.std(scores)
            confidence = max(0, 100 - score_std)  # Higher spread = lower confidence
            
            details = {
                'quality_score': round(quality_score, 2),
                'confidence': round(confidence, 2),
                'input_scores': {
                    'clarity': round(clarity_score, 2),
                    'completeness': round(completeness_score, 2),
                    'context': round(context_score, 2),
                    'structure': round(structure_score, 2)
                }
            }
            
            return quality_score, details
            
        except Exception as e:
            print(f"Error in fuzzy logic calculation: {e}")
            # Fallback to weighted average
            weights = [0.25, 0.25, 0.25, 0.25]
            scores = [clarity_score, completeness_score, context_score, structure_score]
            fallback_score = sum(w * s for w, s in zip(weights, scores))
            
            details = {
                'quality_score': round(fallback_score, 2),
                'confidence': 50.0,  # Lower confidence for fallback
                'input_scores': {
                    'clarity': round(clarity_score, 2),
                    'completeness': round(completeness_score, 2),
                    'context': round(context_score, 2),
                    'structure': round(structure_score, 2)
                },
                'note': 'Fallback calculation used due to fuzzy logic error'
            }
            
            return fallback_score, details

    def calculate_parameter_score(self, text: str, parameter: Dict) -> Tuple[float, Dict]:
        """
        Calculate final score for a parameter using ensemble approach
        """
        # Get scores from different models
        rule_score = self.rule_based_score(text, parameter)
        nlp_score, nlp_details = self.nlp_based_score(text, parameter)
        
        # Calculate completeness and context scores for fuzzy logic
        completeness = len([k for k in parameter['keywords'] if k in text.lower()]) / len(parameter['keywords']) * 100
        context_score = (rule_score + nlp_score) / 2
        
        # Get fuzzy logic score
        fuzzy_score, fuzzy_details = self.fuzzy_logic_score(rule_score, completeness, context_score, nlp_score)
        
        # Weighted ensemble combination
        weights = {
            'rule_based': 0.3,
            'nlp_based': 0.4,
            'fuzzy_logic': 0.3
        }
        
        final_score = (
            weights['rule_based'] * rule_score +
            weights['nlp_based'] * nlp_score +
            weights['fuzzy_logic'] * fuzzy_score
        )
        
        details = {
            'rule_based_score': round(rule_score, 2),
            'nlp_based_score': round(nlp_score, 2),
            'fuzzy_logic_score': round(fuzzy_score, 2),
            'final_score': round(final_score, 2),
            'nlp_details': nlp_details,
            'fuzzy_details': fuzzy_details
        }
        
        return final_score, details

    def calculate_score(
        self,
        clause_analysis: Dict,
        risk_analysis: Dict,
        compliance_scores: Dict,
        semantic_analysis: Dict
    ) -> float:
        """Calculate overall contract score"""
        # Calculate clause coverage score
        coverage_score = clause_analysis['coverage']['essential_clauses']
        
        # Calculate risk score (inverse of risk level)
        risk_score = 1.0 - risk_analysis['metrics']['overall_risk_score']
        
        # Get compliance score
        compliance_score = compliance_scores['overall_compliance_score'] / 100
        
        # Get semantic quality score
        semantic_score = semantic_analysis.get('quality_score', 0.0)
        
        # Calculate weighted score
        weighted_score = sum([
            coverage_score * self.weights['clause_coverage'],
            risk_score * self.weights['risk_score'],
            compliance_score * self.weights['compliance'],
            semantic_score * self.weights['semantic_quality']
        ])
        
        return round(weighted_score * 100, 2)  # Convert to percentage

    def evaluate_contract(self, text: str, parameters: Dict) -> Dict:
        """
        Evaluate entire contract using all parameters
        """
        results = {}
        total_score = 0
        
        for param_name, param_details in parameters.items():
            score, details = self.calculate_parameter_score(text, param_details)
            weighted_score = score * param_details['weight']
            total_score += weighted_score
            
            results[param_name] = {
                'score': round(score, 2),
                'weighted_score': round(weighted_score, 2),
                'details': details
            }
            
        results['total_score'] = round(total_score, 2)
        return results
