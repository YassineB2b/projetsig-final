"""
Storage Service

Handles file storage for uploads and detection results.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, BinaryIO
from werkzeug.utils import secure_filename


class StorageService:
    """Service for managing file storage."""

    def __init__(self, base_path: str):
        """
        Initialize storage service.

        Args:
            base_path: Base directory for file storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        file: BinaryIO,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None
    ) -> Path:
        """
        Save an uploaded file.

        Args:
            file: File object to save.
            filename: Optional custom filename. Auto-generates UUID if not provided.
            subfolder: Optional subfolder within base path.

        Returns:
            Path to saved file (relative to base_path).
        """
        # Generate filename
        if filename:
            safe_filename = secure_filename(filename)
            name, ext = os.path.splitext(safe_filename)
            unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        else:
            # Get extension from original filename if available
            original_name = getattr(file, 'filename', 'image.jpg')
            ext = os.path.splitext(original_name)[1] or '.jpg'
            unique_filename = f"{uuid.uuid4()}{ext}"

        # Create date-based subfolder
        date_folder = datetime.now().strftime("%Y/%m/%d")

        if subfolder:
            save_dir = self.base_path / subfolder / date_folder
        else:
            save_dir = self.base_path / date_folder

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = save_dir / unique_filename

        if hasattr(file, 'save'):
            # Flask FileStorage object
            file.save(str(file_path))
        else:
            # Regular file object
            with open(file_path, 'wb') as f:
                f.write(file.read())

        # Return relative path
        return file_path.relative_to(self.base_path)

    def save_bytes(
        self,
        data: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Path:
        """
        Save raw bytes to file.

        Args:
            data: Bytes to save.
            filename: Filename to use.
            subfolder: Optional subfolder.

        Returns:
            Path to saved file (relative to base_path).
        """
        date_folder = datetime.now().strftime("%Y/%m/%d")

        if subfolder:
            save_dir = self.base_path / subfolder / date_folder
        else:
            save_dir = self.base_path / date_folder

        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / filename

        with open(file_path, 'wb') as f:
            f.write(data)

        return file_path.relative_to(self.base_path)

    def delete_file(self, relative_path: str) -> bool:
        """
        Delete a file.

        Args:
            relative_path: Path relative to base_path.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self.base_path / relative_path

        if file_path.exists():
            file_path.unlink()
            return True

        return False

    def get_full_path(self, relative_path: str) -> Path:
        """Get full path from relative path."""
        return self.base_path / relative_path

    def list_files(
        self,
        subfolder: Optional[str] = None,
        pattern: str = "*"
    ) -> list:
        """
        List files in storage.

        Args:
            subfolder: Optional subfolder to list.
            pattern: Glob pattern for filtering files.

        Returns:
            List of file paths.
        """
        search_path = self.base_path / subfolder if subfolder else self.base_path
        return list(search_path.rglob(pattern))

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            'total_files': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    def cleanup_old_files(self, days: int = 30) -> int:
        """
        Delete files older than specified days.

        Args:
            days: Number of days to retain files.

        Returns:
            Number of files deleted.
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1

        # Clean up empty directories
        self._cleanup_empty_dirs()

        return deleted_count

    def _cleanup_empty_dirs(self):
        """Remove empty directories."""
        for dir_path in sorted(self.base_path.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
