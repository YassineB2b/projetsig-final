"""
REST API Routes

Provides endpoints for license plate detection, history, cameras, and statistics.
"""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename

from app import db, get_pipeline
from app.models.detection import Detection, Camera
from app.services.storage_service import StorageService

api_bp = Blueprint('api', __name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


# ============================================
# Detection Endpoints
# ============================================

@api_bp.route('/detect', methods=['POST'])
def detect_plate():
    """
    Process an uploaded image for license plate detection.

    Request: multipart/form-data with 'image' file
    Response: Detection results with plate numbers and bounding boxes
    """
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': {'code': 'NO_IMAGE', 'message': 'No image file provided'}
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': {'code': 'NO_FILENAME', 'message': 'No file selected'}
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': {'code': 'INVALID_FILE', 'message': 'File type not allowed'}
        }), 400

    # Get pipeline
    pipeline = get_pipeline()
    if pipeline is None:
        return jsonify({
            'success': False,
            'error': {'code': 'PIPELINE_ERROR', 'message': 'ML pipeline not initialized'}
        }), 500

    try:
        # Save uploaded file
        storage = StorageService(current_app.config['UPLOAD_FOLDER'])
        original_path = storage.save_upload(file)

        # Reset file stream pointer after saving
        file.seek(0)

        # Process image
        result = pipeline.process_image(file, save_results=True)

        if result['success']:
            # Save to database
            for plate in result['plates']:
                detection = Detection(
                    plate_number=plate['plate_number'],
                    confidence=plate['detection_confidence'],
                    ocr_confidence=plate['ocr_confidence'],
                    bbox_x1=plate['bbox'][0],
                    bbox_y1=plate['bbox'][1],
                    bbox_x2=plate['bbox'][2],
                    bbox_y2=plate['bbox'][3],
                    original_image_path=str(original_path),
                    cropped_image_path=plate.get('cropped_image_path'),
                    source_type='upload',
                    processing_time_ms=result['processing_time_ms']
                )
                db.session.add(detection)

            db.session.commit()

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"Detection error: {e}")
        return jsonify({
            'success': False,
            'error': {'code': 'PROCESSING_ERROR', 'message': str(e)}
        }), 500


@api_bp.route('/detect/batch', methods=['POST'])
def detect_batch():
    """Process multiple images for license plate detection."""
    if 'images' not in request.files:
        return jsonify({
            'success': False,
            'error': {'code': 'NO_IMAGES', 'message': 'No image files provided'}
        }), 400

    files = request.files.getlist('images')

    if not files:
        return jsonify({
            'success': False,
            'error': {'code': 'EMPTY_LIST', 'message': 'No files in request'}
        }), 400

    pipeline = get_pipeline()
    if pipeline is None:
        return jsonify({
            'success': False,
            'error': {'code': 'PIPELINE_ERROR', 'message': 'ML pipeline not initialized'}
        }), 500

    results = []
    for file in files:
        if file.filename and allowed_file(file.filename):
            result = pipeline.process_image(file, save_results=True)
            result['filename'] = file.filename
            results.append(result)

    return jsonify({
        'success': True,
        'results': results,
        'processed_count': len(results)
    })


# ============================================
# Detection History Endpoints
# ============================================

@api_bp.route('/detections', methods=['GET'])
def get_detections():
    """
    Get detection history with pagination and filtering.

    Query params:
        - page: Page number (default: 1)
        - per_page: Items per page (default: 20, max: 100)
        - start_date: Filter from date (ISO format)
        - end_date: Filter to date (ISO format)
        - plate_search: Search by plate number
        - min_confidence: Minimum confidence threshold
        - source_type: Filter by source (upload, camera)
    """
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    # Build query
    query = Detection.query

    # Date filters
    start_date = request.args.get('start_date')
    if start_date:
        query = query.filter(Detection.created_at >= datetime.fromisoformat(start_date))

    end_date = request.args.get('end_date')
    if end_date:
        query = query.filter(Detection.created_at <= datetime.fromisoformat(end_date))

    # Plate search
    plate_search = request.args.get('plate_search')
    if plate_search:
        query = query.filter(Detection.plate_number.ilike(f'%{plate_search}%'))

    # Confidence filter
    min_confidence = request.args.get('min_confidence', type=float)
    if min_confidence:
        query = query.filter(Detection.confidence >= min_confidence)

    # Source type filter
    source_type = request.args.get('source_type')
    if source_type:
        query = query.filter(Detection.source_type == source_type)

    # Order by newest first
    query = query.order_by(Detection.created_at.desc())

    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'success': True,
        'detections': [d.to_dict() for d in pagination.items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })


@api_bp.route('/detections/<detection_id>', methods=['GET'])
def get_detection(detection_id: str):
    """Get a single detection by ID."""
    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({
            'success': False,
            'error': {'code': 'NOT_FOUND', 'message': 'Detection not found'}
        }), 404

    return jsonify({
        'success': True,
        'detection': detection.to_dict()
    })


@api_bp.route('/detections/<detection_id>', methods=['DELETE'])
def delete_detection(detection_id: str):
    """Delete a detection record."""
    detection = Detection.query.get(detection_id)

    if not detection:
        return jsonify({
            'success': False,
            'error': {'code': 'NOT_FOUND', 'message': 'Detection not found'}
        }), 404

    try:
        # Delete associated files
        if detection.original_image_path:
            Path(detection.original_image_path).unlink(missing_ok=True)
        if detection.cropped_image_path:
            Path(detection.cropped_image_path).unlink(missing_ok=True)

        db.session.delete(detection)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Detection deleted successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': {'code': 'DELETE_ERROR', 'message': str(e)}
        }), 500


# ============================================
# Camera Management Endpoints
# ============================================

@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """List all configured cameras."""
    cameras = Camera.query.all()
    return jsonify({
        'success': True,
        'cameras': [c.to_dict() for c in cameras]
    })


@api_bp.route('/cameras', methods=['POST'])
def add_camera():
    """
    Add a new camera source.

    Request body:
        - name: Camera display name
        - rtsp_url: RTSP stream URL
        - enabled: Whether camera is active (default: true)
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'success': False,
            'error': {'code': 'NO_DATA', 'message': 'No data provided'}
        }), 400

    required = ['name', 'rtsp_url']
    for field in required:
        if field not in data:
            return jsonify({
                'success': False,
                'error': {'code': 'MISSING_FIELD', 'message': f'{field} is required'}
            }), 400

    try:
        camera = Camera(
            name=data['name'],
            rtsp_url=data['rtsp_url'],
            enabled=data.get('enabled', True),
            config=data.get('config', {})
        )
        db.session.add(camera)
        db.session.commit()

        return jsonify({
            'success': True,
            'camera': camera.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': {'code': 'CREATE_ERROR', 'message': str(e)}
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['PUT'])
def update_camera(camera_id: str):
    """Update camera configuration."""
    camera = Camera.query.get(camera_id)

    if not camera:
        return jsonify({
            'success': False,
            'error': {'code': 'NOT_FOUND', 'message': 'Camera not found'}
        }), 404

    data = request.get_json()

    if 'name' in data:
        camera.name = data['name']
    if 'rtsp_url' in data:
        camera.rtsp_url = data['rtsp_url']
    if 'enabled' in data:
        camera.enabled = data['enabled']
    if 'config' in data:
        camera.config = data['config']

    try:
        db.session.commit()
        return jsonify({
            'success': True,
            'camera': camera.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': {'code': 'UPDATE_ERROR', 'message': str(e)}
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id: str):
    """Delete a camera."""
    camera = Camera.query.get(camera_id)

    if not camera:
        return jsonify({
            'success': False,
            'error': {'code': 'NOT_FOUND', 'message': 'Camera not found'}
        }), 404

    try:
        db.session.delete(camera)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'Camera deleted successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': {'code': 'DELETE_ERROR', 'message': str(e)}
        }), 500


# ============================================
# Statistics Endpoints
# ============================================

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics."""
    try:
        # Total detections
        total_detections = Detection.query.count()

        # Today's detections
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_detections = Detection.query.filter(
            Detection.created_at >= today_start
        ).count()

        # Average confidence
        from sqlalchemy import func
        avg_confidence = db.session.query(
            func.avg(Detection.confidence)
        ).scalar() or 0

        # Average processing time
        avg_processing_time = db.session.query(
            func.avg(Detection.processing_time_ms)
        ).scalar() or 0

        # Detections by hour (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(hours=24)
        hourly_data = db.session.query(
            func.date_trunc('hour', Detection.created_at).label('hour'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.created_at >= yesterday
        ).group_by('hour').order_by('hour').all()

        # Detections by day (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        daily_data = db.session.query(
            func.date(Detection.created_at).label('date'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.created_at >= week_ago
        ).group_by('date').order_by('date').all()

        # Active cameras
        active_cameras = Camera.query.filter_by(enabled=True).count()

        return jsonify({
            'success': True,
            'stats': {
                'total_detections': total_detections,
                'today_detections': today_detections,
                'average_confidence': round(avg_confidence, 3),
                'average_processing_time_ms': round(avg_processing_time, 2),
                'active_cameras': active_cameras,
                'detections_by_hour': [
                    {'hour': h.hour.isoformat() if h.hour else None, 'count': h.count}
                    for h in hourly_data
                ],
                'detections_by_day': [
                    {'date': d.date.isoformat() if d.date else None, 'count': d.count}
                    for d in daily_data
                ]
            }
        })

    except Exception as e:
        current_app.logger.error(f"Stats error: {e}")
        return jsonify({
            'success': False,
            'error': {'code': 'STATS_ERROR', 'message': str(e)}
        }), 500


# ============================================
# System Endpoints
# ============================================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint."""
    pipeline = get_pipeline()

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'database': 'unknown',
            'ml_pipeline': 'unknown',
            'gpu': 'unknown'
        }
    }

    # Check database
    try:
        db.session.execute(db.text('SELECT 1'))
        health_status['components']['database'] = 'healthy'
    except Exception:
        health_status['components']['database'] = 'unhealthy'
        health_status['status'] = 'degraded'

    # Check ML pipeline
    if pipeline:
        health_status['components']['ml_pipeline'] = 'healthy'
        system_info = pipeline.get_system_info()
        health_status['gpu_info'] = system_info
        health_status['components']['gpu'] = 'available' if system_info.get('using_gpu') else 'cpu_only'
    else:
        health_status['components']['ml_pipeline'] = 'not_initialized'
        health_status['status'] = 'degraded'

    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@api_bp.route('/settings', methods=['GET'])
def get_settings():
    """Get current system settings."""
    pipeline = get_pipeline()

    settings = {
        'detection_confidence': current_app.config.get('DETECTION_CONFIDENCE', 0.5),
        'max_upload_size_mb': current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024) / (1024 * 1024),
        'storage_retention_days': current_app.config.get('STORAGE_RETENTION_DAYS', 30),
        'use_gpu': current_app.config.get('USE_GPU', True),
        'ocr_languages': current_app.config.get('OCR_LANGUAGES', ['en', 'ar'])
    }

    if pipeline:
        settings.update(pipeline.get_system_info())

    return jsonify({
        'success': True,
        'settings': settings
    })


@api_bp.route('/settings', methods=['PUT'])
def update_settings():
    """Update system settings."""
    data = request.get_json()
    pipeline = get_pipeline()

    if not data:
        return jsonify({
            'success': False,
            'error': {'code': 'NO_DATA', 'message': 'No data provided'}
        }), 400

    updated = []

    # Update confidence threshold
    if 'detection_confidence' in data and pipeline:
        threshold = float(data['detection_confidence'])
        pipeline.set_confidence_threshold(threshold)
        updated.append('detection_confidence')

    return jsonify({
        'success': True,
        'message': f'Updated settings: {", ".join(updated) if updated else "none"}'
    })


# ============================================
# Storage Endpoints
# ============================================

@api_bp.route('/storage/uploads/<path:filename>')
def serve_upload(filename: str):
    """Serve uploaded images."""
    return send_from_directory(
        current_app.config['UPLOAD_FOLDER'],
        filename
    )


@api_bp.route('/storage/detections/<path:filename>')
def serve_detection(filename: str):
    """Serve cropped detection images."""
    return send_from_directory(
        current_app.config['DETECTION_FOLDER'],
        filename
    )
