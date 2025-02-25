"""
Risk analysis module for legal contracts.
Provides comprehensive risk assessment and compliance scoring capabilities.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceCategory(Enum):
    DATA_PRIVACY = "data_privacy"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    REGULATORY = "regulatory"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    SECURITY = "security"
    CONTRACTUAL = "contractual"

@dataclass
class RiskFactor:
    category: str
    level: RiskLevel
    description: str
    impact_score: float  # 0-1
    probability_score: float  # 0-1
    mitigation_suggestions: List[str]

class RiskAnalyzer:
    def __init__(self):
        """Initialize risk analyzer with comprehensive risk patterns"""
        self.risk_patterns = {
            'liability': {
                'patterns': [
                    r'unlimited\s+liability',
                    r'joint\s+and\s+several\s+liability',
                    r'consequential\s+damages?',
                    r'indirect\s+damages?'
                ],
                'level': RiskLevel.HIGH,
                'impact': 0.9,
                'category': 'liability',
                'mitigations': [
                    'Add liability caps',
                    'Include damage limitations',
                    'Specify liability exceptions'
                ]
            },
            'termination': {
                'patterns': [
                    r'immediate\s+termination',
                    r'termination\s+without\s+cause',
                    r'unilateral\s+termination'
                ],
                'level': RiskLevel.MEDIUM,
                'impact': 0.7,
                'category': 'operational',
                'mitigations': [
                    'Add notice periods',
                    'Include cure periods',
                    'Specify termination conditions'
                ]
            },
            'data_privacy': {
                'patterns': [
                    r'personal\s+data',
                    r'data\s+(?:processing|protection)',
                    r'privacy',
                    r'GDPR|CCPA|HIPAA'
                ],
                'level': RiskLevel.HIGH,
                'impact': 0.8,
                'category': 'data_privacy',
                'mitigations': [
                    'Add data protection clauses',
                    'Include breach notification requirements',
                    'Specify data handling procedures'
                ]
            },
            'intellectual_property': {
                'patterns': [
                    r'ownership\s+of\s+(?:IP|intellectual\s+property)',
                    r'(?:patent|copyright|trademark)\s+infringement',
                    r'third[- ]party\s+IP'
                ],
                'level': RiskLevel.HIGH,
                'impact': 0.8,
                'category': 'intellectual_property',
                'mitigations': [
                    'Clear IP ownership clauses',
                    'Add IP indemnification',
                    'Include license terms'
                ]
            }
        }
        
        self.compliance_requirements = {
            ComplianceCategory.DATA_PRIVACY: {
                'required_clauses': [
                    'data protection',
                    'privacy policy',
                    'data breach notification'
                ],
                'prohibited_terms': [
                    'unlimited data usage',
                    'unrestricted data sharing'
                ],
                'weight': 0.9
            },
            ComplianceCategory.INTELLECTUAL_PROPERTY: {
                'required_clauses': [
                    'ip ownership',
                    'license terms',
                    'ip warranties'
                ],
                'prohibited_terms': [
                    'unrestricted ip use',
                    'no ip protection'
                ],
                'weight': 0.8
            },
            ComplianceCategory.REGULATORY: {
                'required_clauses': [
                    'compliance with laws',
                    'regulatory reporting',
                    'audit rights'
                ],
                'prohibited_terms': [
                    'non-compliance',
                    'regulatory violation'
                ],
                'weight': 0.85
            }
        }
        
    def analyze_risks(self, text: str) -> Dict:
        """Perform comprehensive risk analysis of contract text"""
        risks = []
        risk_scores = defaultdict(float)
        
        # Analyze each risk pattern
        for risk_type, risk_info in self.risk_patterns.items():
            matches = []
            for pattern in risk_info['patterns']:
                found = re.finditer(pattern, text, re.IGNORECASE)
                matches.extend([m.group() for m in found])
                
            if matches:
                probability = len(matches) * 0.2  # Increase probability with more matches
                probability = min(1.0, probability)  # Cap at 1.0
                
                risk_factor = RiskFactor(
                    category=risk_info['category'],
                    level=risk_info['level'],
                    description=f"Found {risk_type} risk: {', '.join(matches)}",
                    impact_score=risk_info['impact'],
                    probability_score=probability,
                    mitigation_suggestions=risk_info['mitigations']
                )
                risks.append(risk_factor)
                
                # Calculate risk score
                risk_score = risk_info['impact'] * probability
                risk_scores[risk_type] = risk_score
        
        # Calculate overall risk metrics
        if risks:
            avg_impact = sum(r.impact_score for r in risks) / len(risks)
            avg_probability = sum(r.probability_score for r in risks) / len(risks)
            max_risk_score = max(risk_scores.values())
        else:
            avg_impact = 0.0
            avg_probability = 0.0
            max_risk_score = 0.0
            
        return {
            'risks': [
                {
                    'category': r.category,
                    'level': r.level.value,
                    'description': r.description,
                    'impact_score': r.impact_score,
                    'probability_score': r.probability_score,
                    'mitigation_suggestions': r.mitigation_suggestions
                } for r in risks
            ],
            'risk_scores': dict(risk_scores),
            'metrics': {
                'average_impact': avg_impact,
                'average_probability': avg_probability,
                'max_risk_score': max_risk_score,
                'overall_risk_score': (avg_impact + max_risk_score) / 2
            }
        }
        
    def analyze_compliance(self, text: str) -> Dict:
        """Analyze compliance with various regulatory and contractual requirements"""
        compliance_scores = {}
        missing_requirements = defaultdict(list)
        prohibited_terms_found = defaultdict(list)
        
        for category, requirements in self.compliance_requirements.items():
            category_name = category.value
            required_found = 0
            prohibited_found = 0
            
            # Check required clauses
            for clause in requirements['required_clauses']:
                if re.search(clause, text, re.IGNORECASE):
                    required_found += 1
                else:
                    missing_requirements[category_name].append(clause)
            
            # Check prohibited terms
            for term in requirements['prohibited_terms']:
                if re.search(term, text, re.IGNORECASE):
                    prohibited_found += 1
                    prohibited_terms_found[category_name].append(term)
            
            # Calculate compliance score
            required_score = required_found / len(requirements['required_clauses'])
            prohibited_penalty = prohibited_found * 0.2  # 20% penalty per prohibited term
            
            compliance_scores[category_name] = {
                'score': max(0, (required_score - prohibited_penalty) * 100),
                'weight': requirements['weight'],
                'required_found': required_found,
                'required_total': len(requirements['required_clauses']),
                'prohibited_found': prohibited_found
            }
        
        # Calculate overall compliance score
        weighted_scores = [
            score['score'] * score['weight']
            for score in compliance_scores.values()
        ]
        total_weights = sum(score['weight'] for score in compliance_scores.values())
        overall_score = sum(weighted_scores) / total_weights if total_weights > 0 else 0
        
        return {
            'compliance_scores': compliance_scores,
            'overall_compliance_score': overall_score,
            'missing_requirements': dict(missing_requirements),
            'prohibited_terms_found': dict(prohibited_terms_found)
        }
