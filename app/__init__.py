from flask import Flask
import os
from config import config

def create_app(config_name='default'):
    """Application factory function"""
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)
    
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)  # Initialize the application with config-specific setup

    # Register blueprints
    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Ensure the upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app
