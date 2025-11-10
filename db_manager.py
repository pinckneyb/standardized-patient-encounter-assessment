"""
Database manager for analysis job persistence and resume functionality.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

class AnalysisJobManager:
    """Manages analysis job persistence in PostgreSQL."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def create_job(self, video_filename: str, video_path: str, profile: str, 
                   fps: float, batch_size: int) -> str:
        """Create a new analysis job and return job_id."""
        job_id = str(uuid.uuid4())[:8]
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO analysis_jobs 
                    (job_id, video_filename, video_path, profile, fps, batch_size, 
                     status, current_stage)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending', 'created')
                """, (job_id, video_filename, video_path, profile, fps, batch_size))
                conn.commit()
        
        return job_id
    
    def update_stage(self, job_id: str, stage: str, **kwargs):
        """Update job stage and optionally other fields."""
        set_clauses = ["current_stage = %s", "updated_at = CURRENT_TIMESTAMP"]
        params = [stage]
        
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
    
    def update_progress(self, job_id: str, completed_batches: int, total_batches: int = None):
        """Update batch processing progress."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if total_batches is not None:
                    cur.execute("""
                        UPDATE analysis_jobs 
                        SET completed_batches = %s, total_batches = %s, 
                            progress = ROUND((%s::float / %s) * 100), 
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (completed_batches, total_batches, completed_batches, total_batches, job_id))
                else:
                    cur.execute("""
                        UPDATE analysis_jobs 
                        SET completed_batches = %s,
                            progress = ROUND((%s::float / total_batches) * 100), 
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (completed_batches, completed_batches, job_id))
                conn.commit()
    
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
    
    def mark_error(self, job_id: str, error_message: str):
        """Mark job as failed with error message."""
        self.update_stage(job_id, 'error', status='failed', error_message=error_message)
    
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
