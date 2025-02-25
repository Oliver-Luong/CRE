# Legal Linguistic Comparator

A powerful contract analysis engine that uses NLP and machine learning to analyze legal documents, identify patterns, and provide insights.

## Features

- Contract text analysis and comparison
- Legal pattern matching
- Semantic analysis with GPU acceleration
- Risk analysis and scoring
- Performance metrics tracking
- RESTful API endpoints

## Tech Stack

- Python 3.9+
- Flask
- spaCy
- NLTK
- PyTorch
- Sentence Transformers
- Gunicorn

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /analyze`: Analyze contract text
  ```json
  {
    "contract_text": "Your contract text here..."
  }
  ```

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/Oliver-Luong/CRE.git
   cd CRE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python wsgi.py
   ```

## Deployment on Render

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Use the following settings:
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
   - Environment Variables:
     - `PYTHON_VERSION`: `3.9.0`
     - `PORT`: `8000`

## GPU Support

The application automatically detects and utilizes GPU acceleration when available. To enable GPU support:

1. Install CUDA toolkit and cuDNN
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## License

MIT License
