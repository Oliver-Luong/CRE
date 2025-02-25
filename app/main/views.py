from flask import render_template, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import os
from . import main
from app.analyzer import analyze_document
import logging
import nltk
import spacy
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        logger.info("Starting analysis request")
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                logger.info(f"Saving file to {filepath}")
                file.save(filepath)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                os.remove(filepath)  # Clean up after reading
                logger.info("File successfully read and removed")
            else:
                logger.warning("Invalid file type or no file provided")
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            data = request.get_json()
            if not data:
                logger.error("No JSON data in request")
                return jsonify({'error': 'No data provided'}), 400
                
            text = data.get('text', '')
            if not text or not isinstance(text, str):
                logger.error(f"Invalid text input: {type(text)}")
                return jsonify({'error': 'Invalid text input'}), 400
                
            if not text.strip():
                logger.warning("Empty text provided")
                return jsonify({'error': 'No text provided'}), 400
            
            logger.info(f"Received text input for analysis (length: {len(text)})")

        logger.info("Starting document analysis")
        try:
            results = analyze_document(text)
            logger.info("Analysis completed successfully")
            return jsonify(results)
        except Exception as analysis_error:
            logger.error(f"Analysis error: {str(analysis_error)}", exc_info=True)
            error_msg = str(analysis_error)
            if "Analysis failed:" in error_msg:
                error_msg = error_msg.split("Analysis failed:", 1)[1].strip()
            return jsonify({
                'error': error_msg,
                'details': 'Please check your input and try again.'
            }), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to process request',
            'details': str(e)
        }), 500

@main.route('/system-status')
def system_status():
    try:
        # Test NLTK
        nltk_status = {}
        for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                nltk_status[resource] = "Available"
            except LookupError:
                nltk_status[resource] = "Missing"

        # Test spaCy
        try:
            nlp = spacy.load('en_core_web_sm')
            spacy_status = "Available"
            test_text = "This is a test sentence."
            doc = nlp(test_text)
            spacy_test = "Working"
        except Exception as e:
            spacy_status = f"Error: {str(e)}"
            spacy_test = "Failed"

        return jsonify({
            'status': 'ok',
            'nltk_resources': nltk_status,
            'spacy_model': spacy_status,
            'spacy_test': spacy_test,
            'python_version': sys.version,
            'upload_folder': os.path.exists(current_app.config['UPLOAD_FOLDER'])
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@main.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
