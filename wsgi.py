import os
from app import create_app
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create Flask app
app = create_app(os.getenv('FLASK_CONFIG') or 'default')

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)
