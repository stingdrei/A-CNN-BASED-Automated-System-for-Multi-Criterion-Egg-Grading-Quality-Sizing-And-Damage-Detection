import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional

from fastapi import UploadFile
from fastapi.staticfiles import StaticFiles

from app.config import settings


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
    def get_file_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            return "image"
        raise ValueError(f"Unsupported file type: {ext}")
    
    def is_allowed_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_IMAGE_EXTENSIONS
    
    async def save_file(self, file: UploadFile) -> tuple[str, str]:
        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix.lower()
        stored_name = f"{file_id}{ext}"
        file_path = self.upload_dir / stored_name
        
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError(
                f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit"
            )
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        file_type = self.get_file_type(file.filename)
        return str(file_path), file_type
    
    def save_file_sync(self, file: UploadFile) -> tuple[str, str]:
        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix.lower()
        stored_name = f"{file_id}{ext}"
        file_path = self.upload_dir / stored_name
        
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        total_bytes = 0
        try:
            with open(file_path, "wb") as f:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ValueError(
                            f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit"
                        )
                    f.write(chunk)
            file_type = self.get_file_type(file.filename)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        return str(file_path), file_type
    
    def delete_file(self, file_path: str) -> bool:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception:
            return False


file_service = FileService()