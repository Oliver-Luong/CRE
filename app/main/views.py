from flask import render_template, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import os
from . import main
from app.analyzer import analyze_document

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                os.remove(filepath)  # Clean up after reading
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            text = request.json.get('text', '')
            if not text.strip():
                return jsonify({'error': 'No text provided'}), 400

        results = analyze_document(text)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
