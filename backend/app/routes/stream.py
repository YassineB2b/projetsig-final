"""
Video Streaming Routes

Provides live video feed endpoints with real-time license plate detection.
"""

import cv2
import time
from flask import Blueprint, Response, current_app, jsonify
from app import get_pipeline
from app.services.camera_service import CameraService

stream_bp = Blueprint('stream', __name__)


def generate_frames(camera_id: str):
    """
    Generator function for video streaming with detection overlay.

    Args:
        camera_id: ID of the camera to stream from.

    Yields:
        JPEG frames as multipart response.
    """
    camera_service = CameraService.get_instance()
    pipeline = get_pipeline()

    if not camera_service.is_camera_active(camera_id):
        # Try to start the camera
        success = camera_service.start_camera(camera_id)
        if not success:
            # Return error frame
            error_frame = create_error_frame("Camera not available")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            return

    frame_interval = 1 / 30  # 30 FPS target
    last_detection_time = 0
    detection_interval = 0.5  # Run detection every 0.5 seconds
    last_detections = []

    while True:
        start_time = time.time()

        # Get frame from camera
        frame = camera_service.get_frame(camera_id)

        if frame is None:
            # Camera disconnected or error
            error_frame = create_error_frame("No signal")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
            continue

        # Run detection at intervals (not every frame for performance)
        current_time = time.time()
        if pipeline and (current_time - last_detection_time) >= detection_interval:
            processed_frame, detections = pipeline.process_frame(frame, draw_results=True)
            last_detections = detections
            last_detection_time = current_time
            frame = processed_frame
        else:
            # Draw previous detections on current frame
            frame = draw_cached_detections(frame, last_detections)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Maintain frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


def draw_cached_detections(frame, detections):
    """Draw cached detection results on frame."""
    if not detections:
        return frame

    output = frame.copy()

    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        plate_text = det.get('plate_number', '')
        confidence = det.get('confidence', 0)

        color = (0, 255, 0)  # Green

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{plate_text} ({confidence:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return output


def create_error_frame(message: str):
    """Create an error frame with message."""
    import numpy as np

    # Create black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add error message
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2

    cv2.putText(frame, message, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # Encode as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else b''


@stream_bp.route('/video/<camera_id>')
def video_feed(camera_id: str):
    """
    Live video stream endpoint.

    Returns a multipart JPEG stream for embedding in HTML.
    """
    return Response(
        generate_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@stream_bp.route('/snapshot/<camera_id>')
def snapshot(camera_id: str):
    """
    Get a single frame snapshot from camera.

    Returns a JPEG image.
    """
    camera_service = CameraService.get_instance()

    if not camera_service.is_camera_active(camera_id):
        camera_service.start_camera(camera_id)

    frame = camera_service.get_frame(camera_id)

    if frame is None:
        return jsonify({
            'success': False,
            'error': 'Failed to capture frame'
        }), 500

    ret, buffer = cv2.imencode('.jpg', frame)

    if not ret:
        return jsonify({
            'success': False,
            'error': 'Failed to encode frame'
        }), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')


@stream_bp.route('/webcam')
def webcam_feed():
    """
    Stream from default webcam (device 0).

    For local USB webcam testing.
    """
    return Response(
        generate_webcam_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_webcam_frames():
    """Generate frames from default webcam."""
    cap = cv2.VideoCapture(0)
    pipeline = get_pipeline()

    if not cap.isOpened():
        error_frame = create_error_frame("Webcam not available")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
        return

    last_detection_time = 0
    detection_interval = 0.5
    last_detections = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()
            if pipeline and (current_time - last_detection_time) >= detection_interval:
                processed_frame, detections = pipeline.process_frame(frame, draw_results=True)
                last_detections = detections
                last_detection_time = current_time
                frame = processed_frame
            else:
                frame = draw_cached_detections(frame, last_detections)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

    finally:
        cap.release()
