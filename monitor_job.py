#!/usr/bin/env python3
"""
Job monitoring script - monitors progress of video analysis jobs
"""

import time
import psycopg2
import os
import sys

def monitor_job(job_id: str):
    """Monitor a job and display real-time progress."""
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    start_time = time.time()
    last_progress = -1
    last_stage = ""
    
    print(f"\n{'='*80}")
    print(f"🔍 MONITORING JOB: {job_id}")
    print(f"{'='*80}\n")
    
    try:
        while True:
            elapsed = (time.time() - start_time) / 60
            
            cur = conn.cursor()
            cur.execute("""
                SELECT status, current_stage, progress, error_message, 
                       completed_batches, total_batches, progress_details
                FROM analysis_jobs 
                WHERE job_id = %s
            """, (job_id,))
            
            row = cur.fetchone()
            cur.close()
            
            if not row:
                print(f"❌ Job {job_id} not found")
                break
            
            status, stage, progress, error, completed, total, details = row
            
            # Show updates when something changes
            if progress != last_progress or stage != last_stage:
                timestamp = time.strftime("%H:%M:%S")
                progress_str = f"{progress}%" if progress else "0%"
                batch_str = f"({completed}/{total})" if total else ""
                
                print(f"[{timestamp}] {stage:30s} | {progress_str:5s} {batch_str:15s} | {elapsed:.1f}m elapsed")
                
                if details and details != last_stage:
                    print(f"           └─ {details}")
                
                last_progress = progress
                last_stage = stage
            
            # Check completion
            if status == 'completed':
                print(f"\n{'='*80}")
                print(f"✅ SUCCESS! Job completed in {elapsed:.1f} minutes")
                print(f"{'='*80}\n")
                
                # Show results
                cur = conn.cursor()
                cur.execute("""
                    SELECT pdf_path, storage_key, audio_transcript, enhanced_narrative
                    FROM analysis_jobs WHERE job_id = %s
                """, (job_id,))
                result = cur.fetchone()
                cur.close()
                
                if result:
                    pdf_path, storage_key, transcript, narrative = result
                    print("📊 Results:")
                    print(f"   • PDF: {'✓ Generated' if pdf_path or storage_key else '✗ Missing'}")
                    print(f"   • Transcript: {'✓' if transcript else '✗'} ({len(transcript or '')} chars)")
                    print(f"   • Narrative: {'✓' if narrative else '✗'} ({len(narrative or '')} chars)")
                
                return True
            
            elif status == 'failed':
                print(f"\n{'='*80}")
                print(f"❌ FAILED after {elapsed:.1f} minutes")
                print(f"Error: {error}")
                print(f"{'='*80}\n")
                return False
            
            time.sleep(5)  # Check every 5 seconds
            
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        # Get most recent job
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute("""
            SELECT job_id FROM analysis_jobs 
            ORDER BY created_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        
        if not row:
            print("❌ No jobs found")
            sys.exit(1)
        
        job_id = row[0]
        print(f"📍 Using most recent job: {job_id}")
    
    monitor_job(job_id)
