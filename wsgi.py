"""
WSGI entry point for the Legal Linguistic Comparator application.
"""

from flask import Flask, request, jsonify, Response
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

# Initialize Flask app with optimized settings
app = Flask(__name__)
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
    
    # Use production WSGI server with optimized settings
    from waitress import serve
    
    # Calculate optimal number of threads based on CPU cores
    cpu_count = multiprocessing.cpu_count()
    thread_count = cpu_count * 2  # Rule of thumb: 2 threads per CPU core
    
    logger.info(f"Starting server on port {port} with {thread_count} threads (CPU cores: {cpu_count})")
    serve(
        app,
        host='0.0.0.0',
        port=port,
        threads=thread_count,
        url_scheme='http',
        channel_timeout=30,
        cleanup_interval=30,
        connection_limit=1024,
        max_request_body_size=1024 * 1024 * 10  # 10MB limit
    )
