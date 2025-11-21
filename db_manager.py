"""
Database manager for analysis job persistence and resume functionality.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.error_logger import get_error_logger

class AnalysisJobManager:
    """Manages analysis job persistence in PostgreSQL."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        self.error_logger = get_error_logger()
        self.ensure_schema()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def ensure_schema(self):
        """Ensure the database schema has all required columns for progress tracking."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DO $$ 
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='analysis_jobs' AND column_name='progress_details'
                            ) THEN
                                ALTER TABLE analysis_jobs ADD COLUMN progress_details TEXT;
                            END IF;
                            
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='analysis_jobs' AND column_name='pdf_path'
                            ) THEN
                                ALTER TABLE analysis_jobs ADD COLUMN pdf_path TEXT;
                            END IF;
                            
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='analysis_jobs' AND column_name='output_dir'
                            ) THEN
                                ALTER TABLE analysis_jobs ADD COLUMN output_dir TEXT;
                            END IF;
                            
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='analysis_jobs' AND column_name='last_heartbeat'
                            ) THEN
                                ALTER TABLE analysis_jobs ADD COLUMN last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                            END IF;
                            
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name='analysis_jobs' AND column_name='storage_key'
                            ) THEN
                                ALTER TABLE analysis_jobs ADD COLUMN storage_key TEXT;
                            END IF;
                        END $$;
                    """)
                    conn.commit()
            self.error_logger.log_info("database", None, None, "Schema validated successfully")
        except Exception as e:
            self.error_logger.log_error("database", None, None, 
                                       f"Failed to ensure schema: {str(e)}", e)
    
    def create_job(self, video_filename: str, video_path: str, profile: str, 
                   fps: float, batch_size: int) -> str:
        """Create a new analysis job and return job_id."""
        job_id = str(uuid.uuid4())[:8]
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO analysis_jobs 
                        (job_id, video_filename, video_path, profile, fps, batch_size, 
                         status, current_stage)
                        VALUES (%s, %s, %s, %s, %s, %s, 'pending', 'created')
                    """, (job_id, video_filename, video_path, profile, fps, batch_size))
                    conn.commit()
            
            self.error_logger.log_info("database", job_id, video_filename, 
                                      f"Created job with profile={profile}, fps={fps}, batch_size={batch_size}")
            return job_id
        except Exception as e:
            self.error_logger.log_error("database", job_id, video_filename, 
                                       f"Failed to create job: {str(e)}", e)
            raise
    
    def update_stage(self, job_id: str, stage: str, progress_details: Optional[str] = None, **kwargs):
        """Update job stage and optionally other fields including progress_details."""
        try:
            set_clauses = ["current_stage = %s", "updated_at = CURRENT_TIMESTAMP"]
            params = [stage]
            
            if progress_details is not None:
                set_clauses.append("progress_details = %s")
                params.append(progress_details)
            
            for key, value in kwargs.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            
            params.append(job_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    query = f"""
                        UPDATE analysis_jobs 
                        SET {', '.join(set_clauses)}
                        WHERE job_id = %s
                    """
                    cur.execute(query, params)
                    conn.commit()
            
            self.error_logger.log_info("database", job_id, None, f"Updated stage to: {stage}")
        except Exception as e:
            self.error_logger.log_error("database", job_id, None, 
                                       f"Failed to update stage to {stage}: {str(e)}", e)
            raise
    
    def update_progress(self, job_id: str, completed_batches: int, total_batches: Optional[int] = None):
        """Update batch processing progress."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if total_batches is not None:
                        # Cap progress at 100% (handle cases where actual batches exceed estimate)
                        cur.execute("""
                            UPDATE analysis_jobs 
                            SET completed_batches = %s, total_batches = %s, 
                                progress = LEAST(100, ROUND((%s::float / %s) * 100)), 
                                updated_at = CURRENT_TIMESTAMP
                            WHERE job_id = %s
                        """, (completed_batches, total_batches, completed_batches, total_batches, job_id))
                    else:
                        # Cap progress at 100% even when total_batches is from previous estimate
                        cur.execute("""
                            UPDATE analysis_jobs 
                            SET completed_batches = %s,
                                progress = LEAST(100, ROUND((%s::float / total_batches) * 100)), 
                                updated_at = CURRENT_TIMESTAMP
                            WHERE job_id = %s
                        """, (completed_batches, completed_batches, job_id))
                    conn.commit()
            
            self.error_logger.log_progress("database", job_id, None, 
                                          completed_batches, total_batches or completed_batches,
                                          "Batch processing progress updated")
        except Exception as e:
            self.error_logger.log_error("database", job_id, None, 
                                       f"Failed to update progress: {str(e)}", e)
            raise
    
    def save_audio_transcript(self, job_id: str, transcript: str):
        """Save audio transcript."""
        self.update_stage(job_id, 'audio_transcribed', audio_transcript=transcript)
    
    def save_frame_transcript(self, job_id: str, transcript: str):
        """Save frame analysis transcript."""
        self.update_stage(job_id, 'frames_analyzed', frame_transcript=transcript)
    
    def save_narrative(self, job_id: str, narrative: str):
        """Save enhanced narrative."""
        self.update_stage(job_id, 'narrative_created', enhanced_narrative=narrative)
    
    def save_assessment(self, job_id: str, assessment: str):
        """Save assessment report."""
        self.update_stage(job_id, 'assessment_complete', 
                         assessment_report=assessment, status='completed')
    
    def save_pdf_path(self, job_id: str, pdf_path: str):
        """Save PDF file path."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE analysis_jobs SET pdf_path = %s WHERE job_id = %s",
                    (pdf_path, job_id)
                )
                conn.commit()
    
    def save_output_dir(self, job_id: str, output_dir: str):
        """Save output directory path to database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE analysis_jobs SET output_dir = %s WHERE job_id = %s",
                    (output_dir, job_id)
                )
                conn.commit()
    
    def save_storage_key(self, job_id: str, storage_key: str):
        """Save object storage key for PDF to database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE analysis_jobs SET storage_key = %s WHERE job_id = %s",
                    (storage_key, job_id)
                )
                conn.commit()
    
    def update_heartbeat(self, job_id: str):
        """Update job heartbeat to indicate process is still alive."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE analysis_jobs SET last_heartbeat = CURRENT_TIMESTAMP WHERE job_id = %s",
                        (job_id,)
                    )
                    conn.commit()
        except Exception as e:
            # Don't fail the job if heartbeat update fails - just log it
            self.error_logger.log_error("database", job_id, None, 
                                       f"Failed to update heartbeat: {str(e)}", e)
    
    def mark_error(self, job_id: str, error_message: str):
        """Mark job as failed with error message."""
        try:
            self.update_stage(job_id, 'error', status='failed', error_message=error_message)
            self.error_logger.log_error("database", job_id, None, 
                                       f"Job marked as failed: {error_message}", None)
        except Exception as e:
            self.error_logger.log_error("database", job_id, None, 
                                       f"Failed to mark job as error: {str(e)}", e)
            raise
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details by job_id."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM analysis_jobs WHERE job_id = %s
                """, (job_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    
    def get_incomplete_jobs(self) -> List[Dict[str, Any]]:
        """Get all incomplete jobs that can be resumed."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM analysis_jobs 
                    WHERE status IN ('pending', 'in_progress')
                    ORDER BY created_at DESC
                """)
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all active jobs (queued or in_progress).
        Automatically detects and marks stale jobs (heartbeat >120 min old) as failed.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # First, mark stale jobs as failed
                # A job is stale if last_heartbeat is more than 120 minutes old
                # (Increased from 20min to 120min to support long video processing - 13-min videos take 10-15+ mins)
                cur.execute("""
                    UPDATE analysis_jobs
                    SET status = 'failed',
                        error_message = 'Process died - no heartbeat for >120 minutes',
                        current_stage = 'error'
                    WHERE status = 'in_progress'
                      AND last_heartbeat IS NOT NULL
                      AND last_heartbeat < (CURRENT_TIMESTAMP - INTERVAL '7200 seconds')
                      AND error_message IS NULL
                    RETURNING job_id
                """)
                stale_jobs = cur.fetchall()
                
                if stale_jobs:
                    stale_count = len(stale_jobs)
                    self.error_logger.log_error("stale_job_detection", None, None,
                                               f"Marked {stale_count} stale jobs as failed", None)
                    print(f"⚠️  Detected and marked {stale_count} stale jobs as failed")
                
                conn.commit()
                
                # Now fetch active jobs (excluding the stale ones we just marked)
                cur.execute("""
                    SELECT * FROM analysis_jobs 
                    WHERE status IN ('queued', 'in_progress')
                    ORDER BY created_at DESC
                """)
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent jobs ordered by creation time."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM analysis_jobs 
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def delete_job(self, job_id: str):
        """Delete a job and its data."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analysis_jobs WHERE job_id = %s", (job_id,))
                conn.commit()
    
    def cleanup_temp_files(self, job_id: str):
        """Clean up temporary video files associated with a job."""
        job = self.get_job(job_id)
        if job and job.get('video_path'):
            from pathlib import Path
            video_path = Path(job['video_path'])
            if video_path.exists():
                video_path.unlink()
                print(f"🗑️ Cleaned up video file: {video_path}")
            
            audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
            if audio_path.exists():
                audio_path.unlink()
                print(f"🗑️ Cleaned up audio file: {audio_path}")
    
    def cleanup_old_incomplete_jobs(self):
        """Mark old incomplete jobs as failed and clean up their temp files.
        This runs on app startup to handle any jobs left in-progress from crashes/interruptions.
        Only marks jobs as stale if they haven't been updated in 120+ minutes.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Only mark jobs as stale if they haven't updated in 120+ minutes (support long video processing)
                    cur.execute("""
                        SELECT job_id, video_path, video_filename FROM analysis_jobs 
                        WHERE status IN ('pending', 'in_progress')
                        AND updated_at < NOW() - INTERVAL '120 minutes'
                    """)
                    stale_jobs = cur.fetchall()
                    
                    for job in stale_jobs:
                        job_id = job['job_id']
                        video_filename = job.get('video_filename')
                        
                        cur.execute("""
                            UPDATE analysis_jobs 
                            SET status = 'failed', 
                                error_message = 'Job was interrupted and could not be completed',
                                updated_at = CURRENT_TIMESTAMP
                            WHERE job_id = %s
                        """, (job_id,))
                        
                        self.error_logger.log_warning("cleanup", job_id, video_filename,
                                                     "Stale job marked as failed during cleanup")
                        
                        if job.get('video_path'):
                            from pathlib import Path
                            video_path = Path(job['video_path'])
                            if video_path.exists():
                                try:
                                    video_path.unlink()
                                    self.error_logger.log_info("cleanup", job_id, video_filename,
                                                              f"Cleaned up stale video file: {video_path}")
                                except Exception as e:
                                    self.error_logger.log_error("cleanup", job_id, video_filename,
                                                               f"Could not delete {video_path}: {str(e)}", e)
                            
                            audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
                            if audio_path.exists():
                                try:
                                    audio_path.unlink()
                                    self.error_logger.log_info("cleanup", job_id, video_filename,
                                                              f"Cleaned up stale audio file: {audio_path}")
                                except Exception as e:
                                    self.error_logger.log_error("cleanup", job_id, video_filename,
                                                               f"Could not delete {audio_path}: {str(e)}", e)
                    
                    conn.commit()
                    
                    if stale_jobs:
                        self.error_logger.log_info("cleanup", None, None,
                                                  f"Cleaned up {len(stale_jobs)} stale analysis job(s)")
        except Exception as e:
            self.error_logger.log_error("cleanup", None, None,
                                       f"Failed to cleanup old jobs: {str(e)}", e)
