"""
Database Models

SQLAlchemy models for detections and cameras.
"""

import uuid
from datetime import datetime
from app import db


class Detection(db.Model):
    """License plate detection record."""

    __tablename__ = 'detections'

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    plate_number = db.Column(db.String(50), index=True)
    confidence = db.Column(db.Float)
    ocr_confidence = db.Column(db.Float)

    # Bounding box coordinates
    bbox_x1 = db.Column(db.Integer)
    bbox_y1 = db.Column(db.Integer)
    bbox_x2 = db.Column(db.Integer)
    bbox_y2 = db.Column(db.Integer)

    # Image paths
    original_image_path = db.Column(db.String(500))
    cropped_image_path = db.Column(db.String(500))

    # Source information
    camera_id = db.Column(
        db.String(36),
        db.ForeignKey('cameras.id', ondelete='SET NULL'),
        nullable=True
    )
    source_type = db.Column(
        db.String(20),
        default='upload',
        index=True
    )  # 'upload', 'camera', 'url'

    # Processing metrics
    processing_time_ms = db.Column(db.Float)

    # Timestamps
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        index=True
    )

    # Additional metadata (JSON) - named 'metadata' in DB but 'extra_data' in Python
    # to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    extra_data = db.Column('metadata', db.JSON, default=dict)

    # Relationships
    camera = db.relationship('Camera', backref='detections')

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'plate_number': self.plate_number,
            'confidence': self.confidence,
            'ocr_confidence': self.ocr_confidence,
            'combined_confidence': (
                (self.confidence + self.ocr_confidence) / 2
                if self.confidence and self.ocr_confidence
                else self.confidence or 0
            ),
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'original_image_url': (
                f'/api/storage/uploads/{self.original_image_path}'
                if self.original_image_path else None
            ),
            'cropped_image_url': (
                f'/api/storage/detections/{self.cropped_image_path}'
                if self.cropped_image_path else None
            ),
            'camera_id': self.camera_id,
            'source_type': self.source_type,
            'processing_time_ms': self.processing_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'extra_data': self.extra_data
        }

    def __repr__(self):
        return f'<Detection {self.id}: {self.plate_number}>'


class Camera(db.Model):
    """Camera configuration for live streaming."""

    __tablename__ = 'cameras'

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    name = db.Column(db.String(100), nullable=False)
    rtsp_url = db.Column(db.String(500), nullable=False)
    enabled = db.Column(db.Boolean, default=True, index=True)

    # Status tracking
    last_active_at = db.Column(db.DateTime)
    status = db.Column(
        db.String(20),
        default='disconnected'
    )  # 'connected', 'disconnected', 'error'

    # Configuration (JSON)
    config = db.Column(db.JSON, default=dict)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'rtsp_url': self.rtsp_url,
            'enabled': self.enabled,
            'status': self.status,
            'last_active_at': (
                self.last_active_at.isoformat()
                if self.last_active_at else None
            ),
            'config': self.config,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'stream_url': f'/stream/video/{self.id}'
        }

    def __repr__(self):
        return f'<Camera {self.id}: {self.name}>'


class Setting(db.Model):
    """System settings storage."""

    __tablename__ = 'settings'

    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.JSON, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    @classmethod
    def get(cls, key: str, default=None):
        """Get a setting value."""
        setting = cls.query.get(key)
        return setting.value if setting else default

    @classmethod
    def set(cls, key: str, value):
        """Set a setting value."""
        setting = cls.query.get(key)
        if setting:
            setting.value = value
        else:
            setting = cls(key=key, value=value)
            db.session.add(setting)
        db.session.commit()
        return setting

    def __repr__(self):
        return f'<Setting {self.key}>'
