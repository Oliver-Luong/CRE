"""
Demo script showcasing the comprehensive contract analysis capabilities.
"""

from app.analyzer.contract_analyzer import ContractAnalyzer

def format_analysis_result(result):
    """Format analysis results for display"""
    print("\n=== Contract Analysis Report ===")
    print(f"\nOverall Score: {result.overall_score}%")
    
    print("\n=== Risk Analysis ===")
    for risk in result.risk_analysis['risks']:
        print(f"\nRisk Category: {risk['category']}")
        print(f"Risk Level: {risk['level']}")
        print(f"Impact Score: {risk['impact_score']:.2f}")
        print("Mitigation Suggestions:")
        for suggestion in risk['mitigation_suggestions']:
            print(f"  - {suggestion}")
    
    print("\n=== Compliance Analysis ===")
    for category, scores in result.compliance_scores['compliance_scores'].items():
        print(f"\nCategory: {category}")
        print(f"Score: {scores['score']:.1f}%")
    print(f"\nOverall Compliance Score: {result.compliance_scores['overall_compliance_score']:.1f}%")
    
    print("\n=== Clause Analysis ===")
    coverage = result.clause_analysis['coverage']
    print(f"Essential Clauses Coverage: {coverage['essential_clauses']*100:.1f}%")
    print(f"Total Clauses: {coverage['total_clauses']}")
    
    if result.clause_analysis['missing_clauses']:
        print("\nMissing Essential Clauses:")
        for clause in result.clause_analysis['missing_clauses']:
            print(f"  - {clause}")
    
    if result.clause_analysis['weak_clauses']:
        print("\nWeak Clauses Identified:")
        for clause in result.clause_analysis['weak_clauses']:
            print(f"  - {clause['type']}: {clause['improvement_needed']}")
    
    print("\n=== Performance Metrics ===")
    metrics = result.performance_metrics
    print(f"Execution Time: {metrics.execution_time:.3f} seconds")
    print(f"Analysis Coverage: {metrics.analysis_coverage*100:.1f}%")
    print(f"Confidence Score: {metrics.confidence_score*100:.1f}%")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")

def main():
    # Sample contract text for demonstration
    sample_contract = """
    MASTER SERVICES AGREEMENT
    
    1. Term and Termination
    This Agreement shall commence on the Effective Date and continue for a period of one (1) year.
    Either party may terminate this Agreement upon thirty (30) days written notice.
    
    2. Liability
    The Supplier shall have unlimited liability for any damages arising from this agreement.
    
    3. Data Protection
    Personal data shall be processed in accordance with applicable laws.
    The Supplier shall implement appropriate technical and organizational measures.
    
    4. Intellectual Property
    All intellectual property rights shall remain with their respective owners.
    Each party retains ownership of its pre-existing intellectual property.
    
    5. Payment Terms
    Payment shall be made within 30 days of invoice date.
    Late payments shall incur interest at 1.5% per month.
    """
    
    # Initialize and run analysis
    analyzer = ContractAnalyzer()
    result = analyzer.analyze_contract(sample_contract)
    
    # Display results
    format_analysis_result(result)
    
    # Export results
    analyzer.export_analysis(result, 'contract_analysis_report.json')
    print("\nAnalysis report exported to contract_analysis_report.json")

if __name__ == '__main__':
    main()
