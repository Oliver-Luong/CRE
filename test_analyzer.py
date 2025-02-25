from app.analyzer import analyze_document

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

if __name__ == "__main__":
    test_analyzer()
