"""
Error logging system for video analysis application.
Provides comprehensive error tracking with file rotation and immediate flushing.
"""

import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime
from typing import Optional
import os
from pathlib import Path


class ErrorLogger:
    """
    Singleton error logger that writes all errors to persistent log files.
    Provides immediate flushing and automatic rotation.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log_file = "./logs/video_analysis_errors.log"
        self.logger = logging.getLogger("VideoAnalysisErrorLogger")
        self.logger.setLevel(logging.DEBUG)
        
        Path("./logs").mkdir(exist_ok=True)
        
        self.logger.handlers.clear()
        
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        
        self._initialized = True
        
        self.log_info("system", None, None, "Error logging system initialized")
    
    def log_error(self, stage: str, job_id: Optional[str], video_filename: Optional[str], 
                  error_message: str, exception: Optional[Exception] = None):
        """
        Log an error with full context and traceback.
        
        Args:
            stage: Current processing stage (e.g., 'audio_extraction', 'frame_analysis')
            job_id: Analysis job ID (if available)
            video_filename: Video filename (if available)
            error_message: Error description
            exception: Exception object (if available, will extract traceback)
        """
        job_info = f"job_id={job_id}" if job_id else "job_id=None"
        file_info = f"file={video_filename}" if video_filename else "file=None"
        
        log_message = f"[{stage}] {job_info} | {file_info} | ERROR: {error_message}"
        
        if exception:
            tb = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            log_message += f"\nTraceback:\n{tb}"
        
        self.logger.error(log_message)
        
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_warning(self, stage: str, job_id: Optional[str], video_filename: Optional[str], 
                    warning_message: str):
        """
        Log a warning message.
        
        Args:
            stage: Current processing stage
            job_id: Analysis job ID (if available)
            video_filename: Video filename (if available)
            warning_message: Warning description
        """
        job_info = f"job_id={job_id}" if job_id else "job_id=None"
        file_info = f"file={video_filename}" if video_filename else "file=None"
        
        log_message = f"[{stage}] {job_info} | {file_info} | WARNING: {warning_message}"
        
        self.logger.warning(log_message)
        
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_info(self, stage: str, job_id: Optional[str], video_filename: Optional[str], 
                 info_message: str):
        """
        Log an informational message (e.g., stage entry/exit, progress milestones).
        
        Args:
            stage: Current processing stage
            job_id: Analysis job ID (if available)
            video_filename: Video filename (if available)
            info_message: Information message
        """
        job_info = f"job_id={job_id}" if job_id else "job_id=None"
        file_info = f"file={video_filename}" if video_filename else "file=None"
        
        log_message = f"[{stage}] {job_info} | {file_info} | INFO: {info_message}"
        
        self.logger.info(log_message)
        
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_progress(self, stage: str, job_id: Optional[str], video_filename: Optional[str],
                     current: int, total: int, message: str = ""):
        """
        Log progress milestone.
        
        Args:
            stage: Current processing stage
            job_id: Analysis job ID (if available)
            video_filename: Video filename (if available)
            current: Current progress count
            total: Total count
            message: Optional additional message
        """
        percent = (current / total * 100) if total > 0 else 0
        progress_msg = f"Progress: {current}/{total} ({percent:.1f}%)"
        if message:
            progress_msg += f" - {message}"
        
        self.log_info(stage, job_id, video_filename, progress_msg)
    
    def log_stage_entry(self, stage: str, job_id: Optional[str], video_filename: Optional[str]):
        """Log entry into a processing stage."""
        self.log_info(stage, job_id, video_filename, f"Entering stage: {stage}")
    
    def log_stage_exit(self, stage: str, job_id: Optional[str], video_filename: Optional[str], 
                       success: bool = True):
        """Log exit from a processing stage."""
        status = "SUCCESS" if success else "FAILED"
        self.log_info(stage, job_id, video_filename, f"Exiting stage: {stage} - {status}")
    
    def get_log_file_path(self) -> str:
        """Get the current log file path."""
        return self.log_file


_error_logger_instance = ErrorLogger()


def get_error_logger() -> ErrorLogger:
    """Get the singleton ErrorLogger instance."""
    return _error_logger_instance
