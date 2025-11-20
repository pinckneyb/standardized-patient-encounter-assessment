"""
Object Storage Manager for persistent PDF storage using Replit Object Storage.
"""

from replit.object_storage import Client
import os
from pathlib import Path
from typing import Optional
from utils.error_logger import get_error_logger

class StorageManager:
    """Manages PDF uploads and downloads to Replit Object Storage."""
    
    def __init__(self, bucket_name: str = "ai-video-analysis"):
        """
        Initialize storage manager.
        
        Args:
            bucket_name: Name of the bucket to use for storage
        """
        self.bucket_name = bucket_name
        self.client = Client()
        self.error_logger = get_error_logger()
    
    def upload_pdf(self, pdf_path: str, job_id: str, video_filename: str) -> Optional[str]:
        """
        Upload PDF to object storage.
        
        Args:
            pdf_path: Local path to PDF file
            job_id: Analysis job ID
            video_filename: Original video filename
            
        Returns:
            Object storage key (path) if successful, None otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                self.error_logger.log_error(
                    "storage_upload", job_id, video_filename,
                    f"PDF file not found: {pdf_path}", None
                )
                return None
            
            # Create storage key: pdfs/{job_id}/report.pdf
            base_name = Path(video_filename).stem
            storage_key = f"pdfs/{base_name}_{job_id}/report.pdf"
            
            # Upload file to object storage
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                self.client.upload_from_bytes(storage_key, pdf_bytes)
            
            file_size = len(pdf_bytes)
            self.error_logger.log_info(
                "storage_upload", job_id, video_filename,
                f"PDF uploaded to object storage: {storage_key} ({file_size:,} bytes)"
            )
            print(f"✅ PDF uploaded to object storage: {storage_key} ({file_size:,} bytes)")
            
            return storage_key
            
        except Exception as e:
            self.error_logger.log_error(
                "storage_upload", job_id, video_filename,
                f"Failed to upload PDF to object storage: {str(e)}", e
            )
            print(f"❌ Failed to upload PDF to object storage: {e}")
            return None
    
    def download_pdf(self, storage_key: str, local_path: str) -> bool:
        """
        Download PDF from object storage to local file.
        
        Args:
            storage_key: Object storage key (path)
            local_path: Local path to save PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download file from object storage
            pdf_bytes = self.client.download_as_bytes(storage_key)
            
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(pdf_bytes)
            
            print(f"✅ PDF downloaded from object storage: {storage_key} -> {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download PDF from object storage: {e}")
            return False
    
    def get_pdf_bytes(self, storage_key: str) -> Optional[bytes]:
        """
        Get PDF bytes directly from object storage.
        
        Args:
            storage_key: Object storage key (path)
            
        Returns:
            PDF bytes if successful, None otherwise
        """
        try:
            return self.client.download_as_bytes(storage_key)
        except Exception as e:
            print(f"❌ Failed to get PDF from object storage: {e}")
            return None
    
    def delete_pdf(self, storage_key: str) -> bool:
        """
        Delete PDF from object storage.
        
        Args:
            storage_key: Object storage key (path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(storage_key)
            print(f"✅ PDF deleted from object storage: {storage_key}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete PDF from object storage: {e}")
            return False
    
    def list_pdfs(self, prefix: str = "pdfs/") -> list[str]:
        """
        List all PDFs in object storage.
        
        Args:
            prefix: Prefix to filter keys
            
        Returns:
            List of storage keys
        """
        try:
            keys = self.client.list()
            return [str(key) for key in keys if str(key).startswith(prefix)]
        except Exception as e:
            print(f"❌ Failed to list PDFs from object storage: {e}")
            return []
