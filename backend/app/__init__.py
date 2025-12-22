"""
License Plate Recognition System - Flask Application Factory

Initializes the Flask app with all extensions, blueprints, and ML pipeline.
"""

import os
import logging
from pathlib import Path
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()

# Global ML pipeline instance
lpr_pipeline = None


def create_app(config_name: str = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_name: Configuration to use ('development', 'production', 'testing').

    Returns:
        Configured Flask application.
    """
    # Determine config
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    # Create Flask app
    app = Flask(
        __name__,
        template_folder='../../frontend/templates',
        static_folder='../../frontend/static'
    )

    # Load configuration
    from app.config import config_by_name
    app.config.from_object(config_by_name[config_name])

    # Setup logging
    setup_logging(app)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize ML pipeline
    init_ml_pipeline(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Create storage directories
    create_storage_dirs(app)

    app.logger.info(f"LPR Application initialized in {config_name} mode")

    return app


def setup_logging(app: Flask) -> None:
    """Configure application logging."""
    log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Reduce noise from some libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def init_ml_pipeline(app: Flask) -> None:
    """Initialize the ML pipeline."""
    global lpr_pipeline

    from ml.pipeline import LPRPipeline

    try:
        model_path = app.config.get('DETECTOR_MODEL_PATH')
        confidence = app.config.get('DETECTION_CONFIDENCE', 0.5)
        storage_path = app.config.get('DETECTION_FOLDER')

        lpr_pipeline = LPRPipeline(
            detector_model_path=model_path,
            confidence_threshold=confidence,
            use_gpu=app.config.get('USE_GPU', True),
            storage_path=storage_path
        )

        app.lpr_pipeline = lpr_pipeline
        app.logger.info("ML Pipeline initialized successfully")

    except Exception as e:
        app.logger.error(f"Failed to initialize ML pipeline: {e}")
        app.lpr_pipeline = None


def register_blueprints(app: Flask) -> None:
    """Register Flask blueprints."""
    from app.routes.api import api_bp
    from app.routes.views import views_bp
    from app.routes.stream import stream_bp

    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(views_bp)
    app.register_blueprint(stream_bp, url_prefix='/stream')


def register_error_handlers(app: Flask) -> None:
    """Register error handlers."""
    from flask import jsonify

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': {'code': 'BAD_REQUEST', 'message': str(error)}
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': {'code': 'NOT_FOUND', 'message': 'Resource not found'}
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': {'code': 'INTERNAL_ERROR', 'message': 'An internal error occurred'}
        }), 500


def create_storage_dirs(app: Flask) -> None:
    """Create required storage directories."""
    dirs = [
        app.config.get('UPLOAD_FOLDER', 'storage/uploads'),
        app.config.get('DETECTION_FOLDER', 'storage/detections'),
        app.config.get('LOG_FOLDER', 'storage/logs')
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_pipeline():
    """Get the global ML pipeline instance."""
    global lpr_pipeline
    return lpr_pipeline
