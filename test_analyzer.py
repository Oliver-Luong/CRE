from app.analyzer import analyze_document
import unittest
from app.analyzer.risk_analyzer import RiskAnalyzer, RiskLevel, ComplianceCategory
from app.analyzer.performance_metrics import PerformanceAnalyzer, PerformanceMetrics

# Sample contract text for testing
sample_contract = """
CONSULTING SERVICES AGREEMENT

This Agreement is made and entered into on February 25, 2025, by and between:

ABC Consulting Inc., a corporation organized under the laws of California ("Consultant")
and
XYZ Corp., a Delaware corporation ("Client").

1. SERVICES
Consultant agrees to provide strategic business consulting services ("Services") as detailed in Exhibit A.

2. COMPENSATION
Client shall pay Consultant $150 per hour for Services rendered. Payment terms are net 30 days.

3. TERM AND TERMINATION
This Agreement shall commence on March 1, 2025 and continue for 12 months.
Either party may terminate this Agreement with 30 days written notice.

4. CONFIDENTIALITY
Consultant shall maintain strict confidentiality of all Client information.

5. INTELLECTUAL PROPERTY
All work product created by Consultant shall be owned by Client.

6. INDEMNIFICATION
Each party shall indemnify and hold harmless the other party against all claims.

7. GOVERNING LAW
This Agreement shall be governed by California law.

8. MODIFICATION
Any modification to this Agreement must be in writing and signed by both parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.

ABC Consulting Inc.                XYZ Corp.
_____________________            _____________________
By: John Smith                   By: Jane Doe
Title: CEO                       Title: President
"""

def test_analyzer():
    try:
        # Run the analysis
        results = analyze_document(sample_contract)
        
        # Print overall score
        print(f"\nOverall Contract Score: {results['score']}/100\n")
        
        # Print detailed analysis for each category
        print("Detailed Analysis:")
        print("=" * 50)
        
        for category, analysis in results['analysis'].items():
            print(f"\n{category}")
            print("-" * len(category))
            print(f"Score: {analysis['score']}/100")
            print("Details:", analysis['details'])
            print("Key Findings:", analysis['findings'])
        
        # Print recommendations
        print("\nRecommendations:")
        print("=" * 50)
        for rec in results['recommendations']:
            print(f"- {rec}")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

class TestRiskAnalyzer(unittest.TestCase):
    def setUp(self):
        self.risk_analyzer = RiskAnalyzer()
        
    def test_risk_analysis(self):
        test_text = """
        The Supplier shall have unlimited liability for any damages arising from this agreement.
        Personal data shall be processed in accordance with applicable laws.
        Either party may terminate this agreement without cause upon written notice.
        """
        
        results = self.risk_analyzer.analyze_risks(test_text)
        
        self.assertIn('risks', results)
        self.assertIn('risk_scores', results)
        self.assertIn('metrics', results)
        
        # Verify high-risk patterns were detected
        risk_types = [risk['category'] for risk in results['risks']]
        self.assertIn('liability', risk_types)
        self.assertIn('data_privacy', risk_types)
        
        # Verify risk metrics
        self.assertGreater(results['metrics']['overall_risk_score'], 0)
        
    def test_compliance_analysis(self):
        test_text = """
        1. Data Protection
        All personal data shall be processed in accordance with the privacy policy.
        
        2. Intellectual Property
        All intellectual property rights shall remain with the respective owners.
        """
        
        results = self.risk_analyzer.analyze_compliance(test_text)
        
        self.assertIn('compliance_scores', results)
        self.assertIn('overall_compliance_score', results)
        self.assertGreater(results['overall_compliance_score'], 0)

class TestPerformanceAnalyzer(unittest.TestCase):
    def setUp(self):
        self.perf_analyzer = PerformanceAnalyzer()
        
    def test_performance_measurement(self):
        start_time = self.perf_analyzer.start_measurement()
        
        # Add a small delay to ensure measurable execution time
        import time
        time.sleep(0.1)
        
        # Simulate some analysis
        analysis_results = {
            'clauses': [
                {'text': 'clause 1', 'confidence_score': 0.8},
                {'text': 'clause 2', 'confidence_score': 0.9}
            ]
        }
        
        metrics = self.perf_analyzer.calculate_metrics(start_time, analysis_results)
        
        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.analysis_coverage, 0)
        self.assertGreater(metrics.confidence_score, 0)
        
    def test_validation(self):
        analysis_results = {
            'clauses': [
                {'text': 'clause 1', 'type': 'liability'},
                {'text': 'clause 2', 'type': 'termination'}
            ]
        }
        
        ground_truth = {
            'clauses': [
                {'text': 'clause 1', 'type': 'liability'},
                {'text': 'clause 2', 'type': 'compliance'}
            ]
        }
        
        validation = self.perf_analyzer.validate_analysis(analysis_results, ground_truth)
        
        self.assertGreaterEqual(validation.precision, 0)
        self.assertGreaterEqual(validation.recall, 0)
        self.assertGreaterEqual(validation.f1_score, 0)
        self.assertIn('per_category', validation.details)
        
    def test_metrics_tracking(self):
        metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage=100.0,
            analysis_coverage=0.8,
            confidence_score=0.9
        )
        
        self.perf_analyzer.track_metrics(metrics)
        summary = self.perf_analyzer.get_metrics_summary()
        
        self.assertIn('execution_time', summary)
        self.assertIn('coverage', summary)
        self.assertIn('confidence', summary)
        self.assertEqual(summary['sample_size'], 1)

if __name__ == "__main__":
    unittest.main()
