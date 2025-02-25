import os
from app import create_app
from waitress import serve
from paste.translogger import TransLogger
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
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# Create Flask app
try:
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    logger.info("Flask application created successfully")
except Exception as e:
    logger.error(f"Failed to create Flask application: {str(e)}")
    sys.exit(1)

if __name__ == '__main__':
    try:
        logger.info("Starting Waitress server...")
        serve(
            TransLogger(app, setup_console_handler=True),
            host='127.0.0.1',
            port=8080,
            threads=6,
            channel_timeout=300,
            cleanup_interval=30
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
