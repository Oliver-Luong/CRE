"""
Advanced legal pattern recognition module for contract analysis.
Provides sophisticated pattern matching and legal clause identification.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import spacy
from .ml_models import TextProcessor

class ClauseType(Enum):
    """Types of legal clauses"""
    DEFINITION = "definition"
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    TERMINATION = "termination"
    INDEMNIFICATION = "indemnification"
    WARRANTY = "warranty"
    LIMITATION = "limitation"
    GOVERNING_LAW = "governing_law"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    AMENDMENT = "amendment"
    NOTICE = "notice"
    FORCE_MAJEURE = "force_majeure"

@dataclass
class ClauseAnalysis:
    """Analysis results for a legal clause"""
    type: ClauseType
    text: str
    strength: float
    importance_level: float
    risk_level: float
    complexity_score: float
    key_terms: List[str]
    dependencies: List[str]

class LegalPatternMatcher:
    """Identifies and analyzes legal clauses and patterns"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.nlp = spacy.load('en_core_web_sm')
    
    def analyze_clause(self, text: str) -> Optional[ClauseAnalysis]:
        """Analyze a clause and determine its characteristics"""
        doc = self.nlp(text)
        
        # Identify clause type
        clause_type = self._identify_clause_type(doc)
        if not clause_type:
            return None
            
        # Extract key terms
        key_terms = self._extract_key_terms(doc)
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies(doc)
        
        # Calculate metrics
        strength = self._calculate_strength(doc)
        importance = self._calculate_importance(doc, clause_type)
        risk = self._calculate_risk(doc, clause_type)
        complexity = self._calculate_complexity(doc)
        
        return ClauseAnalysis(
            type=clause_type,
            text=text,
            strength=strength,
            importance_level=importance,
            risk_level=risk,
            complexity_score=complexity,
            key_terms=key_terms,
            dependencies=dependencies
        )
    
    def analyze_contract_structure(self, text: str) -> Dict:
        """Analyze the structural elements of a contract"""
        doc = self.nlp(text)
        sections = self._identify_sections(doc)
        
        # Analyze presence and quality of key sections
        section_scores = {}
        for section, content in sections.items():
            section_scores[section] = {
                'present': bool(content),
                'quality': self._calculate_section_quality(content) if content else 0.0,
                'completeness': self._calculate_section_completeness(content) if content else 0.0
            }
        
        # Calculate overall structural score
        total_score = sum(s['quality'] * s['completeness'] for s in section_scores.values())
        avg_score = total_score / len(section_scores) if section_scores else 0.0
        
        return {
            'sections': section_scores,
            'structure_score': avg_score,
            'missing_sections': [s for s, data in section_scores.items() if not data['present']]
        }
    
    def _identify_clause_type(self, doc) -> Optional[ClauseType]:
        """Identify the type of legal clause"""
        text = doc.text.lower()
        
        # Map keywords to clause types
        type_patterns = {
            ClauseType.DEFINITION: ['means', 'defined', 'shall mean', 'refers to'],
            ClauseType.OBLIGATION: ['shall', 'must', 'will', 'agrees to'],
            ClauseType.PROHIBITION: ['shall not', 'may not', 'prohibited', 'restricted'],
            ClauseType.PERMISSION: ['may', 'permitted', 'allowed', 'authorized'],
            ClauseType.TERMINATION: ['terminate', 'termination', 'cancel', 'end'],
            # Add more patterns for other clause types
        }
        
        for clause_type, patterns in type_patterns.items():
            if any(pattern in text for pattern in patterns):
                return clause_type
        
        return None
    
    def _extract_key_terms(self, doc) -> List[str]:
        """Extract key legal terms from the clause"""
        return self.text_processor.extract_phrases(doc.text)
    
    def _analyze_dependencies(self, doc) -> List[str]:
        """Identify dependencies on other clauses or sections"""
        dependencies = []
        # Add dependency analysis logic
        return dependencies
    
    def _calculate_strength(self, doc) -> float:
        """Calculate the strength of the clause language"""
        # Add strength calculation logic
        return 0.5
    
    def _calculate_importance(self, doc, clause_type: ClauseType) -> float:
        """Calculate the importance level of the clause"""
        # Add importance calculation logic
        return 0.5
    
    def _calculate_risk(self, doc, clause_type: ClauseType) -> float:
        """Calculate the risk level associated with the clause"""
        # Add risk calculation logic
        return 0.5
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate the complexity score of the clause"""
        # Add complexity calculation logic
        return 0.5
    
    def _identify_sections(self, doc) -> Dict[str, str]:
        """Identify major contract sections"""
        # Add section identification logic
        return {}
    
    def _calculate_section_quality(self, text: str) -> float:
        """Calculate the quality score for a section"""
        # Add section quality calculation logic
        return 0.5
    
    def _calculate_section_completeness(self, text: str) -> float:
        """Calculate the completeness score for a section"""
        # Add section completeness calculation logic
        return 0.5
