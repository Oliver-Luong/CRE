# Legal Contract Analyzer

A sophisticated web application for analyzing legal documents, providing insights into document quality, structure, and compliance. The application uses natural language processing and machine learning techniques to evaluate legal texts.

## Features

- **Document Analysis**
  - Similarity scoring with legal corpus
  - Legal jargon density calculation
  - Key phrase extraction
  - Contract structure analysis
  - Readability metrics

- **User Interface**
  - Modern, intuitive web interface
  - Multiple input methods (direct text, file upload, drag & drop)
  - Real-time analysis
  - Interactive results display
  - Export functionality

## Technology Stack

- **Backend**
  - Python 3.8+
  - Flask (Web Framework)
  - NLTK (Natural Language Processing)
  - SpaCy (Advanced NLP)
  - scikit-learn (Machine Learning)
  - Waitress (Production WSGI Server)

- **Frontend**
  - HTML5
  - CSS3
  - JavaScript (ES6+)
  - Modern UI/UX design

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-contract-analyzer
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. Download SpaCy model:
```bash
python -m spacy download en_core_web_sm
```

6. Create a .env file:
```
FLASK_CONFIG=production
SECRET_KEY=your-super-secret-key-change-this
FLASK_APP=wsgi.py
```

## Running the Application

### Development
```bash
flask run
```

### Production
```bash
# Windows (Waitress)
waitress-serve --call 'wsgi:create_app'

# Unix/Linux (Gunicorn)
gunicorn wsgi:app
```

The application will be available at `http://localhost:8080` (or your configured port).

## Usage

1. Access the web interface through your browser
2. Input legal text using one of the following methods:
   - Type directly in the editor
   - Paste text from clipboard
   - Upload a document file
   - Drag and drop a file
3. Click "Analyze" to process the document
4. View the comprehensive analysis results
5. Export results if needed

## API Endpoints

- `GET /`: Main application interface
- `POST /analyze`: Document analysis endpoint
  - Accepts: JSON with text field or multipart form data with file
  - Returns: JSON with analysis results

## Development

The project follows a modular Flask application structure:
```
legal-contract-analyzer/
├── app/                    # Application package
│   ├── __init__.py        # App factory
│   ├── analyzer/          # Analysis module
│   └── main/              # Main views
├── static/                # Static files
├── templates/             # HTML templates
├── uploads/              # Upload directory
├── config.py             # Configuration
├── wsgi.py              # WSGI entry point
└── requirements.txt     # Dependencies
```

## Deployment

The application is ready for deployment on various platforms:

### Render
1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn wsgi:app`

### Heroku
1. Create a new Heroku app
2. Connect your GitHub repository
3. Add Python buildpack
4. Deploy from GitHub

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Contact

[Your Contact Information]
