from flask import Flask, render_template, request, jsonify
from Script import (
    preprocess_text, compute_similarity, calculate_jargon_density,
    extract_phrases_with_spacy, analyze_contract_structure,
    analyze_readability, calculate_final_score, legal_jargon
)

app = Flask(__name__)

# Default repository - in production, this should be loaded from files
repository = [
    "The plaintiff filed a lawsuit for breach of contract.",
    "An affidavit was presented in court to support the case.",
    "Jurisdiction is determined by the geographical boundaries of the court."
]

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
        structure_score = analyze_contract_structure(input_document)
        readability_score = analyze_readability(input_document)
        
        final_score = calculate_final_score(
            similarity_score,
            jargon_density,
            len(phrases),
            structure_score,
            readability_score
        )

        return jsonify({
            'similarity_score': round(similarity_score, 2),
            'jargon_density': round(jargon_density, 2),
            'key_phrases': phrases,
            'structure_score': round(structure_score, 2),
            'readability_score': round(readability_score, 2),
            'final_score': round(final_score, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
