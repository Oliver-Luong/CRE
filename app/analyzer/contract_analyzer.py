"""
Comprehensive contract analysis module that integrates risk analysis,
compliance validation, and performance tracking.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from .risk_analyzer import RiskAnalyzer
from .performance_metrics import PerformanceAnalyzer, PerformanceMetrics
from .legal_patterns import LegalPatternMatcher, ClauseType
from .scoring_models import ContractScorer
from .ml_models import SemanticAnalyzer, TextProcessor

@dataclass
class AnalysisResult:
    """Comprehensive analysis result including all metrics and insights"""
    risk_analysis: Dict
    compliance_scores: Dict
    performance_metrics: PerformanceMetrics
    semantic_analysis: Dict
    clause_analysis: Dict
    overall_score: float
    timestamp: datetime
    recommendations: List[str]

class ContractAnalyzer:
    """
    Comprehensive contract analyzer that combines risk analysis,
    compliance validation, and performance tracking.
    """
    
    def __init__(self):
        self._text_processor = None
        self._risk_analyzer = None
        self._perf_analyzer = None
        self._pattern_matcher = None
        self._contract_scorer = None
        self._semantic_analyzer = None
    
    @property
    def text_processor(self):
        if self._text_processor is None:
            self._text_processor = TextProcessor()
        return self._text_processor
    
    @property
    def risk_analyzer(self):
        if self._risk_analyzer is None:
            self._risk_analyzer = RiskAnalyzer()
        return self._risk_analyzer
    
    @property
    def perf_analyzer(self):
        if self._perf_analyzer is None:
            self._perf_analyzer = PerformanceAnalyzer()
        return self._perf_analyzer
    
    @property
    def pattern_matcher(self):
        if self._pattern_matcher is None:
            self._pattern_matcher = LegalPatternMatcher()
        return self._pattern_matcher
    
    @property
    def contract_scorer(self):
        if self._contract_scorer is None:
            self._contract_scorer = ContractScorer()
        return self._contract_scorer
    
    @property
    def semantic_analyzer(self):
        if self._semantic_analyzer is None:
            self._semantic_analyzer = SemanticAnalyzer()
        return self._semantic_analyzer
    
    def analyze_contract(self, contract_text: str) -> AnalysisResult:
        """
        Perform comprehensive contract analysis including risk assessment,
        compliance validation, and performance tracking.
        """
        # Start performance tracking
        start_time = self.perf_analyzer.start_measurement()
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(contract_text)
        
        try:
            # 1. Risk Analysis
            risk_analysis = self.risk_analyzer.analyze_risks(processed_text)
            
            # 2. Semantic Analysis
            semantic_analysis = self.semantic_analyzer.analyze_semantics(contract_text)
            
            # 3. Clause Analysis
            clause_analysis = self.pattern_matcher.analyze_contract_structure(contract_text)
            
            # 4. Calculate overall score
            overall_score = self.contract_scorer.calculate_score(
                clause_analysis=clause_analysis,
                risk_analysis=risk_analysis,
                compliance_scores=risk_analysis['compliance'],
                semantic_analysis=semantic_analysis
            )
            
            # 5. Generate recommendations
            recommendations = self._generate_recommendations(
                risk_analysis,
                semantic_analysis,
                clause_analysis
            )
            
            # Stop performance tracking
            perf_metrics = self.perf_analyzer.stop_measurement(start_time)
            
            return AnalysisResult(
                risk_analysis=risk_analysis,
                compliance_scores=risk_analysis['compliance'],
                performance_metrics=perf_metrics,
                semantic_analysis=semantic_analysis,
                clause_analysis=clause_analysis,
                overall_score=overall_score,
                timestamp=datetime.now(),
                recommendations=recommendations
            )
            
        except Exception as e:
            # Stop performance tracking even if analysis fails
            self.perf_analyzer.stop_measurement(start_time)
            raise
    
    def _generate_recommendations(
        self,
        risk_analysis: Dict,
        semantic_analysis: Dict,
        clause_analysis: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_analysis['metrics']['overall_risk_score'] > 0.7:
            recommendations.append(
                "High risk detected. Consider legal review of identified risk areas."
            )
        
        # Semantic-based recommendations
        if semantic_analysis['scores']['clarity'] < 0.5:
            recommendations.append(
                "Low clarity score. Consider simplifying complex sentences and legal jargon."
            )
        
        # Structure-based recommendations
        missing_sections = clause_analysis.get('missing_sections', [])
        if missing_sections:
            recommendations.append(
                f"Missing important sections: {', '.join(missing_sections)}."
            )
        
        return recommendations
