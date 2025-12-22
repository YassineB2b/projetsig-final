"""
Camera Service

Manages camera connections for live video streaming.
Supports RTSP streams, USB webcams, and IP cameras.
"""

import cv2
import threading
import time
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CameraStream:
    """Individual camera stream handler."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize camera stream.

        Args:
            camera_id: Unique camera identifier.
            source: Video source (RTSP URL, device ID, or file path).
            reconnect_delay: Seconds to wait before reconnection attempt.
        """
        self.camera_id = camera_id
        self.source = source
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[any] = None
        self.frame_lock = threading.Lock()

        self.is_running = False
        self.is_connected = False
        self.last_frame_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start the camera stream."""
        if self.is_running:
            return True

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait for initial connection
        timeout = 5.0
        start_time = time.time()
        while not self.is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        return self.is_connected

    def stop(self):
        """Stop the camera stream."""
        self.is_running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.is_connected = False

    def get_frame(self):
        """Get the latest frame."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        while self.is_running:
            try:
                # Connect if needed
                if self.cap is None or not self.cap.isOpened():
                    self._connect()

                if not self.is_connected:
                    time.sleep(self.reconnect_delay)
                    continue

                # Read frame
                ret, frame = self.cap.read()

                if ret:
                    with self.frame_lock:
                        self.frame = frame
                    self.last_frame_time = datetime.now()
                    self.error_message = None
                else:
                    # Failed to read frame
                    logger.warning(f"Camera {self.camera_id}: Failed to read frame")
                    self.is_connected = False
                    self._disconnect()
                    time.sleep(self.reconnect_delay)

            except Exception as e:
                logger.error(f"Camera {self.camera_id} error: {e}")
                self.error_message = str(e)
                self.is_connected = False
                self._disconnect()
                time.sleep(self.reconnect_delay)

    def _connect(self):
        """Connect to camera source."""
        logger.info(f"Connecting to camera {self.camera_id}: {self.source}")

        try:
            # Determine source type
            if isinstance(self.source, int) or self.source.isdigit():
                # USB webcam by device ID
                source = int(self.source) if isinstance(self.source, str) else self.source
            else:
                # RTSP URL or file path
                source = self.source

            # Create capture with backend selection
            if isinstance(source, str) and source.startswith('rtsp://'):
                # RTSP stream - use FFMPEG backend
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                # Set buffer size for RTSP
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap = cv2.VideoCapture(source)

            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if self.cap.isOpened():
                # Read a test frame
                ret, _ = self.cap.read()
                if ret:
                    self.is_connected = True
                    self.error_message = None
                    logger.info(f"Camera {self.camera_id} connected successfully")
                else:
                    raise Exception("Could not read initial frame")
            else:
                raise Exception("Could not open video source")

        except Exception as e:
            self.is_connected = False
            self.error_message = str(e)
            logger.error(f"Camera {self.camera_id} connection failed: {e}")

    def _disconnect(self):
        """Disconnect from camera source."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_status(self) -> dict:
        """Get camera status information."""
        return {
            'camera_id': self.camera_id,
            'source': self.source,
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'last_frame_time': (
                self.last_frame_time.isoformat()
                if self.last_frame_time else None
            ),
            'error_message': self.error_message
        }


class CameraService:
    """
    Singleton service for managing multiple camera streams.
    """

    _instance: Optional['CameraService'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize camera service."""
        self.cameras: Dict[str, CameraStream] = {}
        self._db_synced = False

    @classmethod
    def get_instance(cls) -> 'CameraService':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def sync_from_database(self):
        """Load cameras from database configuration."""
        if self._db_synced:
            return

        try:
            from app.models.detection import Camera
            from app import db

            cameras = Camera.query.filter_by(enabled=True).all()

            for cam in cameras:
                if cam.id not in self.cameras:
                    self.add_camera(cam.id, cam.rtsp_url, auto_start=False)

            self._db_synced = True
            logger.info(f"Synced {len(cameras)} cameras from database")

        except Exception as e:
            logger.error(f"Failed to sync cameras from database: {e}")

    def add_camera(
        self,
        camera_id: str,
        source: str,
        auto_start: bool = True
    ) -> bool:
        """
        Add a new camera.

        Args:
            camera_id: Unique camera identifier.
            source: Video source (RTSP URL, device ID, etc.).
            auto_start: Whether to start streaming immediately.

        Returns:
            True if camera was added successfully.
        """
        if camera_id in self.cameras:
            logger.warning(f"Camera {camera_id} already exists")
            return False

        camera = CameraStream(camera_id, source)
        self.cameras[camera_id] = camera

        if auto_start:
            camera.start()

        logger.info(f"Added camera: {camera_id}")
        return True

    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera."""
        if camera_id not in self.cameras:
            return False

        camera = self.cameras.pop(camera_id)
        camera.stop()

        logger.info(f"Removed camera: {camera_id}")
        return True

    def start_camera(self, camera_id: str) -> bool:
        """Start a camera stream."""
        # Try to load from database if not in memory
        if camera_id not in self.cameras:
            try:
                from app.models.detection import Camera

                cam = Camera.query.get(camera_id)
                if cam:
                    self.add_camera(cam.id, cam.rtsp_url, auto_start=False)
            except Exception as e:
                logger.error(f"Failed to load camera {camera_id}: {e}")
                return False

        if camera_id not in self.cameras:
            return False

        return self.cameras[camera_id].start()

    def stop_camera(self, camera_id: str) -> bool:
        """Stop a camera stream."""
        if camera_id not in self.cameras:
            return False

        self.cameras[camera_id].stop()
        return True

    def get_frame(self, camera_id: str):
        """Get latest frame from a camera."""
        if camera_id not in self.cameras:
            return None

        return self.cameras[camera_id].get_frame()

    def is_camera_active(self, camera_id: str) -> bool:
        """Check if camera is active and connected."""
        if camera_id not in self.cameras:
            return False

        camera = self.cameras[camera_id]
        return camera.is_running and camera.is_connected

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """Get status of a specific camera."""
        if camera_id not in self.cameras:
            return None

        return self.cameras[camera_id].get_status()

    def get_all_statuses(self) -> list:
        """Get status of all cameras."""
        return [cam.get_status() for cam in self.cameras.values()]

    def start_all(self):
        """Start all cameras."""
        for camera in self.cameras.values():
            camera.start()

    def stop_all(self):
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()

    def add_webcam(self, device_id: int = 0) -> str:
        """
        Add default webcam as a camera source.

        Args:
            device_id: Webcam device ID (default: 0).

        Returns:
            Camera ID.
        """
        camera_id = f"webcam_{device_id}"
        self.add_camera(camera_id, device_id)
        return camera_id
