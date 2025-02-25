from flask import Flask, render_template, request, jsonify
from Script import (
    preprocess_text, compute_similarity, calculate_jargon_density,
    extract_phrases_with_spacy, analyze_contract_structure,
    analyze_readability, calculate_final_score, legal_jargon
)
import os

app = Flask(__name__)

# Default repository - in production, this should be loaded from files
repository = [
    "The plaintiff filed a lawsuit for breach of contract.",
    "An affidavit was presented in court to support the case.",
    "Jurisdiction is determined by the geographical boundaries of the court."
]

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        input_document = request.json.get('text', '')
        if not input_document.strip():
            return jsonify({'error': 'Empty document provided'}), 400

        # Process and analyze the document
        preprocessed_input = preprocess_text(input_document)
        preprocessed_repository = [preprocess_text(doc) for doc in repository]
        
        similarity_score = compute_similarity(preprocessed_repository, preprocessed_input)
        jargon_density = calculate_jargon_density(preprocessed_input, legal_jargon)
        phrases = extract_phrases_with_spacy(input_document)
        structure_analysis = analyze_contract_structure(input_document)
        readability_score = analyze_readability(input_document)
        
        final_score = calculate_final_score(
            similarity_score,
            jargon_density,
            len(phrases),
            structure_analysis,
            readability_score
        )

        # Normalize jargon_density for display (lower is better)
        normalized_jargon_score = max(0, 100 - jargon_density * 2)

        # Create recommendations based on scores
        recommendations = []
        
        # Basic metrics recommendations
        if similarity_score < 70:
            recommendations.append("Consider reviewing similar legal documents to improve consistency")
        if normalized_jargon_score < 70:
            recommendations.append("The contract contains high legal jargon density. Consider simplifying the language")
        if readability_score < 70:
            recommendations.append("The text could be more readable. Consider using shorter sentences and simpler language")

        # Legal parameter recommendations
        for param, details in structure_analysis['details'].items():
            if details['score'] < 70:
                if param == 'assent_mutuality':
                    recommendations.append("Strengthen the expression of mutual assent and clear offer/acceptance")
                elif param == 'definiteness_completeness':
                    recommendations.append("Add more specific details about essential terms (price, duration, scope, duties)")
                elif param == 'consideration_bargain':
                    recommendations.append("Clarify the exchange of value or consideration between parties")
                elif param == 'integration_consistency':
                    recommendations.append("Consider adding a clear integration/merger clause")
                elif param == 'modification_termination':
                    recommendations.append("Add clear provisions for contract modification and termination")
                elif param == 'risk_remedies':
                    recommendations.append("Include specific remedies and dispute resolution procedures")
                elif param == 'compliance_standards':
                    recommendations.append("Strengthen compliance with legal standards and regulations")
                elif param == 'language_clarity':
                    recommendations.append("Improve clarity by defining key terms and using consistent language")

        return jsonify({
            'score': round(final_score, 2),
            'analysis': {
                'similarity': round(similarity_score, 2),
                'jargon_density': round(normalized_jargon_score, 2),
                'readability': round(readability_score, 2),
                'legal_parameters': structure_analysis['details'],
                'key_phrases': phrases
            },
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
