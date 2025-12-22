-- License Plate Recognition System
-- Database Schema Initialization

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Detections table
CREATE TABLE IF NOT EXISTS detections (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    plate_number VARCHAR(50),
    confidence DECIMAL(5,4),
    ocr_confidence DECIMAL(5,4),
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    original_image_path VARCHAR(500),
    cropped_image_path VARCHAR(500),
    camera_id VARCHAR(36),
    source_type VARCHAR(20) DEFAULT 'upload',
    processing_time_ms DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    name VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(500) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    status VARCHAR(20) DEFAULT 'disconnected',
    last_active_at TIMESTAMP WITH TIME ZONE,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Settings table
CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add foreign key constraint
ALTER TABLE detections
    ADD CONSTRAINT fk_detections_camera
    FOREIGN KEY (camera_id)
    REFERENCES cameras(id)
    ON DELETE SET NULL;

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_plate_number ON detections(plate_number);
CREATE INDEX IF NOT EXISTS idx_detections_camera_id ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_source_type ON detections(source_type);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);
CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras(enabled);

-- Insert default settings
INSERT INTO settings (key, value) VALUES
    ('detection_confidence_threshold', '"0.5"'),
    ('max_upload_size_mb', '"16"'),
    ('storage_retention_days', '"30"'),
    ('use_gpu', '"true"'),
    ('ocr_languages', '["en", "ar"]')
ON CONFLICT (key) DO NOTHING;

-- Create function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for cameras table
DROP TRIGGER IF EXISTS update_cameras_updated_at ON cameras;
CREATE TRIGGER update_cameras_updated_at
    BEFORE UPDATE ON cameras
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for settings table
DROP TRIGGER IF EXISTS update_settings_updated_at ON settings;
CREATE TRIGGER update_settings_updated_at
    BEFORE UPDATE ON settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
