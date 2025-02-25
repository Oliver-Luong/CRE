"""
WSGI entry point for the Legal Linguistic Comparator application.
"""

from app import create_app
from cachelib import SimpleCache
from app.analyzer.contract_analyzer import ContractAnalyzer
import logging
from datetime import datetime
import json
import multiprocessing
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = create_app()
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order for faster responses
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'http'

# Initialize cache with larger capacity
cache = SimpleCache(default_timeout=3600, threshold=10000)

# Initialize analyzer as a global singleton lazily
_analyzer = None
_analyzer_lock = multiprocessing.Lock()

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = ContractAnalyzer()
    return _analyzer

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_contract():
    """
    Analyze a contract document with caching and streaming response.
    
    Expected JSON payload:
    {
        "contract_text": "Full text of the contract..."
    }
    """
    try:
        data = request.get_json()
        if not data or 'contract_text' not in data:
            return jsonify({
                'error': 'Missing contract_text in request body'
            }), 400
            
        contract_text = data['contract_text']
        if not contract_text.strip():
            return jsonify({
                'error': 'Empty contract text provided'
            }), 400
        
        # Check cache first
        cache_key = f"analysis_{hash(contract_text)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return Response(
                json.dumps(cached_result),
                mimetype='application/json'
            )
            
        # Get analyzer instance and perform analysis
        analyzer = get_analyzer()
        result = analyzer.analyze_contract(contract_text)
        
        # Cache the result
        cache.set(cache_key, result)
        
        return Response(
            json.dumps(result),
            mimetype='application/json'
        )
            
    except Exception as e:
        logger.error(f"Error analyzing contract: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
