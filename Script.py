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
from app.analyzer.scoring_models import ContractScorer

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy language model
nlp = spacy.load('en_core_web_sm')

# Initialize the contract scorer
contract_scorer = ContractScorer()

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
    Analyzes the structural elements and legal parameters of a contract.
    Returns a detailed analysis of various legal aspects.
    """
    # Define key sections and their indicators with weights
    legal_parameters = {
        'assent_mutuality': {
            'keywords': [
                'offer', 'accept', 'agree', 'mutual', 'consent', 'meeting of minds', 'intention',
                'parties agree', 'hereby agrees', 'mutual understanding'
            ],
            'weight': 0.15
        },
        'definiteness_completeness': {
            'keywords': [
                'price', 'duration', 'scope', 'duties', 'obligations', 'terms', 'conditions',
                'payment terms', 'delivery schedule', 'performance metrics', 'specifications'
            ],
            'weight': 0.15
        },
        'consideration_bargain': {
            'keywords': [
                'consideration', 'exchange', 'payment', 'compensation', 'value', 'benefit',
                'detriment', 'in exchange for', 'in return for', 'mutual promises'
            ],
            'weight': 0.10
        },
        'integration_consistency': {
            'keywords': [
                'entire agreement', 'integration', 'merger', 'complete agreement', 'final agreement',
                'supersedes', 'prior agreements', 'sole agreement', 'complete understanding'
            ],
            'weight': 0.10
        },
        'modification_termination': {
            'keywords': [
                'amendment', 'modify', 'terminate', 'revoke', 'cancel', 'waiver',
                'termination rights', 'modification procedure', 'notice period'
            ],
            'weight': 0.10
        },
        'risk_remedies': {
            'keywords': [
                'damages', 'breach', 'arbitration', 'mediation', 'dispute', 'force majeure',
                'indemnification', 'liability', 'limitation of liability', 'remedies'
            ],
            'weight': 0.15
        },
        'compliance_standards': {
            'keywords': [
                'law', 'regulation', 'statute', 'compliance', 'ucc', 'legal', 'jurisdiction',
                'governing law', 'applicable law', 'regulatory requirements'
            ],
            'weight': 0.15
        },
        'language_clarity': {
            'keywords': [
                'defined', 'means', 'shall mean', 'refers to', 'interpretation',
                'definitions', 'defined terms', 'construction', 'herein'
            ],
            'weight': 0.10
        }
    }

    # Use the new scoring system
    analysis_results = contract_scorer.evaluate_contract(text, legal_parameters)
    
    # Format the results for the existing API
    formatted_results = {
        'score': analysis_results['total_score'],
        'details': {
            param: {
                'score': details['score'],
                'status': 'Present' if details['score'] > 70 else 'Missing or Insufficient',
                'analysis': details['details']
            }
            for param, details in analysis_results.items()
            if param != 'total_score'
        }
    }
    
    return formatted_results

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
def calculate_final_score(similarity_score, jargon_density, num_phrases, structure_analysis, readability_score):
    """
    Combines all metrics into a weighted final score with error handling.
    All input scores should be on a 0-100 scale.
    """
    try:
        # Normalize jargon_density to a 0-100 scale where lower density is better
        normalized_jargon_score = max(0, 100 - jargon_density * 2)  # * 2 because density over 50% is very high
        
        # Extract structure score from detailed analysis
        structure_score = structure_analysis['score']
        
        weights = {
            'similarity': 0.15,  # Reduced weight to accommodate new parameters
            'jargon': 0.10,
            'readability': 0.15,
            'structure': 0.60,  # Increased weight for legal parameters
        }
        
        final_score = (
            weights['similarity'] * similarity_score +
            weights['jargon'] * normalized_jargon_score +
            weights['readability'] * readability_score +
            weights['structure'] * structure_score
        )
        
        # Ensure final score is between 0 and 100
        return max(0, min(100, final_score))
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