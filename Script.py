# Import required libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import string
import numpy as np
from flask import Flask, render_template, request, jsonify
from difflib import SequenceMatcher

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy language model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess_text(text):
    """
    Tokenizes and preprocesses input text by:
    - Lowercasing
    - Removing stopwords and punctuation
    """
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

# Extract key phrases using SpaCy
def extract_phrases_with_spacy(text):
    """
    Uses SpaCy to extract noun chunks as key phrases.
    """
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    return phrases

# Compute similarity using TF-IDF
def compute_similarity(repository, input_document):
    """
    Computes cosine similarity between the input document and a repository of legal documents.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(repository)
    input_vector = vectorizer.transform([input_document])
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    return np.mean(similarities) * 100

# Calculate legal jargon density
def calculate_jargon_density(text, legal_jargon):
    """
    Calculates the density of legal jargon in the text.
    """
    tokens = word_tokenize(text)
    jargon_count = sum(1 for token in tokens if token in legal_jargon)
    return (jargon_count / len(tokens)) * 100 if tokens else 0

# Enhanced legal jargon dictionary
legal_jargon = {
    # Common legal terms
    "plaintiff", "defendant", "affidavit", "jurisdiction", "contract",
    "whereas", "hereinafter", "pursuant", "notwithstanding", "forthwith",
    "indemnify", "liability", "termination", "breach", "covenant",
    "warranty", "arbitration", "consideration", "force majeure", "governing law",
    
    # Contract-specific terms
    "agreement", "parties", "obligations", "terms", "conditions",
    "confidentiality", "intellectual property", "compensation", "termination",
    "severability", "assignment", "modification", "waiver", "notices",
    "representations"
}

# Add structure analysis
def analyze_contract_structure(text):
    """
    Analyzes the structural elements of a contract.
    Returns a score based on presence of key contract sections.
    """
    key_sections = {
        'parties': ['between', 'party', 'parties'],
        'recitals': ['whereas', 'background', 'recitals'],
        'definitions': ['means', 'defined', 'definitions'],
        'obligations': ['shall', 'must', 'agrees to'],
        'term': ['term', 'duration', 'period'],
        'termination': ['terminate', 'termination', 'end'],
        'governing_law': ['govern', 'jurisdiction', 'applicable law'],
        'signatures': ['signed', 'executed', 'agreed']
    }
    
    doc = nlp(text.lower())
    sections_found = 0
    
    for section, keywords in key_sections.items():
        if any(keyword in doc.text for keyword in keywords):
            sections_found += 1
    
    return (sections_found / len(key_sections)) * 100

def analyze_readability(text):
    """
    Analyzes the readability of the contract using basic metrics.
    """
    sentences = [sent.text for sent in nlp(text).sents]
    if not sentences:
        return 0
    
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    long_words = sum(1 for word in words if len(word) > 6)
    long_word_ratio = long_words / len(words) if words else 0
    
    # Lower scores mean better readability
    readability_score = (avg_sentence_length * 0.5 + long_word_ratio * 100 * 0.5)
    return max(0, 100 - readability_score)

# Update the final score calculation
def calculate_final_score(similarity_score, jargon_density, num_phrases, structure_score, readability_score):
    """
    Combines all metrics into a weighted final score with error handling.
    """
    try:
        weights = {
            'similarity': 0.3,
            'jargon': 0.2,
            'phrases': 0.1,
            'structure': 0.25,
            'readability': 0.15
        }
        
        return (
            weights['similarity'] * similarity_score +
            weights['jargon'] * jargon_density +
            weights['phrases'] * min(100, num_phrases * 5) +  # Cap phrase contribution
            weights['structure'] * structure_score +
            weights['readability'] * readability_score
        )
    except Exception as e:
        print(f"Error calculating final score: {e}")
        return 0

# Analyze document function
def analyze_document(text):
    # Perform comprehensive analysis
    similarity_score = compute_similarity(["This is a sample contract"], text)
    jargon_density = calculate_jargon_density(text, legal_jargon)
    num_phrases = len(extract_phrases_with_spacy(text))
    structure_score = analyze_contract_structure(text)
    readability_score = analyze_readability(text)
    
    final_score = calculate_final_score(similarity_score, jargon_density, num_phrases, structure_score, readability_score)
    
    # Format the response
    response = {
        'overall_score': final_score,
        'assent_and_mutuality': {
            'score': 0,
            'elements': {}
        },
        'completeness': {
            'score': 0,
            'missing_terms': []
        },
        'consideration': {
            'score': 0,
            'clauses': []
        },
        'integration': {
            'score': 0,
            'has_integration_clause': False
        },
        'risk_allocation': {
            'score': 0,
            'missing_elements': []
        },
        'modification': {
            'score': 0,
            'missing_elements': []
        },
        'compliance': {
            'score': 0,
            'missing_elements': []
        },
        'unconscionability': {
            'score': 0,
            'warnings': []
        },
        'readability': readability_score,
        'recommendations': []
    }
    
    return response

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for analyzing legal text"""
    try:
        data = request.get_json()
        text = data.get('original_text', '').strip()

        if not text:
            return jsonify({
                'error': 'Text is required for analysis'
            }), 400

        # Perform analysis
        result = analyze_document(text)
        
        if 'error' in result:
            return jsonify({
                'error': result['error']
            }), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)