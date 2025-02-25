"""
Performance metrics and validation module for contract analysis.
Provides tools for measuring and validating analysis quality.
"""

from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    execution_time: float
    memory_usage: float
    analysis_coverage: float
    confidence_score: float
    validation_score: Optional[float] = None

@dataclass
class ValidationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    details: Dict

class PerformanceAnalyzer:
    def __init__(self):
        """Initialize the performance analyzer"""
        self.metrics_history = defaultdict(list)
        self.baseline_metrics = None
        
    def start_measurement(self) -> float:
        """Start measuring execution time"""
        return time.time()
        
    def calculate_metrics(self, start_time: float, analysis_results: Dict) -> PerformanceMetrics:
        """Calculate performance metrics for an analysis run"""
        execution_time = time.time() - start_time
        
        # Calculate analysis coverage
        total_sections = len(analysis_results.get('clauses', []))
        analyzed_sections = sum(1 for clause in analysis_results.get('clauses', [])
                              if clause.get('confidence_score', 0) > 0.5)
        coverage = analyzed_sections / total_sections if total_sections > 0 else 0
        
        # Calculate confidence score
        confidence_scores = [
            clause.get('confidence_score', 0)
            for clause in analysis_results.get('clauses', [])
        ]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=0.0,  # Placeholder for actual memory measurement
            analysis_coverage=coverage,
            confidence_score=avg_confidence
        )
        
    def validate_analysis(self, analysis_results: Dict, ground_truth: Dict) -> ValidationResult:
        """Validate analysis results against ground truth data"""
        # Initialize counters for each metric
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        
        # Validate clause classification
        predicted_clauses = {
            clause['text']: clause['type']
            for clause in analysis_results.get('clauses', [])
        }
        actual_clauses = {
            clause['text']: clause['type']
            for clause in ground_truth.get('clauses', [])
        }
        
        for text, pred_type in predicted_clauses.items():
            actual_type = actual_clauses.get(text)
            if actual_type:
                if pred_type == actual_type:
                    true_positives[pred_type] += 1
                else:
                    false_positives[pred_type] += 1
                    false_negatives[actual_type] += 1
            else:
                false_positives[pred_type] += 1
                
        for text, actual_type in actual_clauses.items():
            if text not in predicted_clauses:
                false_negatives[actual_type] += 1
        
        # Calculate metrics
        total_tp = sum(true_positives.values())
        total_fp = sum(false_positives.values())
        total_fn = sum(false_negatives.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-category metrics
        category_metrics = {}
        for category in set(list(true_positives.keys()) + 
                          list(false_positives.keys()) + 
                          list(false_negatives.keys())):
            tp = true_positives[category]
            fp = false_positives[category]
            fn = false_negatives[category]
            
            cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            
            category_metrics[category] = {
                'precision': cat_precision,
                'recall': cat_recall,
                'f1_score': cat_f1
            }
        
        return ValidationResult(
            accuracy=(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0,
            precision=precision,
            recall=recall,
            f1_score=f1,
            details={
                'per_category': category_metrics,
                'confusion_matrix': {
                    'true_positives': dict(true_positives),
                    'false_positives': dict(false_positives),
                    'false_negatives': dict(false_negatives)
                }
            }
        )
        
    def track_metrics(self, metrics: PerformanceMetrics, category: str = 'default'):
        """Track metrics over time"""
        self.metrics_history[category].append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
    def get_metrics_summary(self, category: str = 'default') -> Dict:
        """Get summary statistics for tracked metrics"""
        if not self.metrics_history[category]:
            return {}
            
        metrics_list = self.metrics_history[category]
        execution_times = [m['metrics'].execution_time for m in metrics_list]
        coverage_scores = [m['metrics'].analysis_coverage for m in metrics_list]
        confidence_scores = [m['metrics'].confidence_score for m in metrics_list]
        
        return {
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'coverage': {
                'mean': np.mean(coverage_scores),
                'std': np.std(coverage_scores),
                'min': np.min(coverage_scores),
                'max': np.max(coverage_scores)
            },
            'confidence': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'sample_size': len(metrics_list)
        }
        
    def set_baseline_metrics(self, metrics: PerformanceMetrics):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
        
    def compare_to_baseline(self, current_metrics: PerformanceMetrics) -> Dict:
        """Compare current metrics to baseline"""
        if not self.baseline_metrics:
            return {}
            
        return {
            'execution_time_change': (
                (current_metrics.execution_time - self.baseline_metrics.execution_time)
                / self.baseline_metrics.execution_time
            ),
            'coverage_change': (
                (current_metrics.analysis_coverage - self.baseline_metrics.analysis_coverage)
                / self.baseline_metrics.analysis_coverage
            ),
            'confidence_change': (
                (current_metrics.confidence_score - self.baseline_metrics.confidence_score)
                / self.baseline_metrics.confidence_score
            )
        }
