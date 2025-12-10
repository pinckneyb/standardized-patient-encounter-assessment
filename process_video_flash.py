#!/usr/bin/env python3
"""
Flash 2.5 Video Processing Worker - Simplified background processor using Google Gemini Flash.
Replaces the OpenAI frame-by-frame approach with native video ingestion via Flash 2.5.

Usage: python process_video_flash.py <job_id>
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import traceback


def get_output_dir(video_filename: str, job_id: str) -> Path:
    """
    Get the output directory for a specific video analysis job.
    Creates the directory if it doesn't exist.
    """
    base_name = Path(video_filename).stem
    output_dir = Path('outputs') / f"{base_name}_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_analysis_file(content: str, category: str, video_filename: str, job_id: str) -> str:
    """
    Save analysis results to per-job output folder.
    """
    output_dir = get_output_dir(video_filename, job_id)
    
    filename_map = {
        'transcript': 'transcript.txt',
        'narrative': 'narrative.txt',
        'assessment': 'assessment.txt'
    }
    
    filename = filename_map.get(category, f'{category}.txt')
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)


def process_video_flash(job_id: str):
    """
    Process a video analysis job using Google Flash 2.5 for video analysis.
    
    Stages:
    1. extracting_audio - Extract audio track from video
    2. transcribing_audio - Transcribe audio using Whisper API
    3. uploading_video - Upload video to Google for Flash 2.5 analysis
    4. analyzing_video - Analyze video with Flash 2.5
    5. generating_assessment - Generate medical assessment
    6. generating_pdf - Create PDF report
    7. completed - All done
    
    Args:
        job_id: The analysis job ID to process
    """
    from db_manager import AnalysisJobManager
    from google_flash_client import GoogleFlashClient
    from audio_utils import extract_audio_from_video, transcribe_audio_with_whisper
    from pdf_generator import create_assessment_pdf
    from storage_manager import StorageManager
    from utils.error_logger import get_error_logger
    
    error_logger = get_error_logger()
    db = AnalysisJobManager()
    flash_client = None
    audio_path = None
    
    job = db.get_job(job_id)
    if not job:
        print(f"❌ Job {job_id} not found in database")
        return
    
    video_filename = job['video_filename']
    video_path = job['video_path']
    profile = job['profile']
    
    print(f"🚀 Starting Flash 2.5 processing for job {job_id}")
    print(f"   Video: {video_filename}")
    print(f"   Profile: {profile}")
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        error_msg = "GOOGLE_API_KEY not found in environment"
        print(f"❌ {error_msg}")
        db.mark_error(job_id, error_msg)
        return
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        error_msg = "OPENAI_API_KEY not found in environment (needed for Whisper transcription)"
        print(f"❌ {error_msg}")
        db.mark_error(job_id, error_msg)
        return
    
    try:
        db.update_stage(job_id, 'in_progress', status='in_progress',
                       progress_details="Starting Flash 2.5 video analysis")
        error_logger.log_stage_entry("flash_processing", job_id, video_filename)
        
        # ========================================
        # Phase 1: Extract Audio
        # ========================================
        print("\n" + "="*60)
        print("🎤 Phase 1: Audio Extraction")
        print("="*60)
        
        db.update_stage(job_id, 'extracting_audio', 
                       progress_details="Extracting audio track from video")
        db.update_heartbeat(job_id)
        
        audio_path = extract_audio_from_video(video_path)
        
        # ========================================
        # Phase 2: Transcribe Audio
        # ========================================
        audio_transcript = ""
        if audio_path:
            print("\n" + "="*60)
            print("🗣️ Phase 2: Audio Transcription")
            print("="*60)
            
            db.update_stage(job_id, 'transcribing_audio', 
                           progress_details="Transcribing audio using Whisper API")
            db.update_heartbeat(job_id)
            
            audio_transcript = transcribe_audio_with_whisper(audio_path, openai_api_key)
            
            if audio_transcript:
                db.save_audio_transcript(job_id, audio_transcript)
                
                transcript_file = save_analysis_file(
                    audio_transcript, 'transcript', video_filename, job_id
                )
                print(f"📄 Transcript saved: {transcript_file}")
            else:
                print("⚠️ No audio transcript generated")
        else:
            print("⚠️ No audio found in video (skipping transcription)")
        
        # ========================================
        # Phase 3: Upload Video to Google
        # ========================================
        print("\n" + "="*60)
        print("📤 Phase 3: Uploading Video to Google")
        print("="*60)
        
        db.update_stage(job_id, 'uploading_video', 
                       progress_details="Uploading video to Google servers")
        db.update_heartbeat(job_id)
        
        flash_client = GoogleFlashClient(api_key=google_api_key)
        
        def progress_callback(message: str, percent: int):
            db.update_stage(job_id, 'uploading_video', progress_details=message)
            db.update_heartbeat(job_id)
            print(f"   {message} ({percent}%)")
        
        flash_client.upload_video(video_path, progress_callback=progress_callback)
        
        # ========================================
        # Phase 4: Analyze Video with Flash 2.5
        # ========================================
        print("\n" + "="*60)
        print("🤖 Phase 4: Analyzing Video with Flash 2.5")
        print("="*60)
        
        db.update_stage(job_id, 'analyzing_video', 
                       progress_details="Analyzing video with Flash 2.5 AI")
        db.update_heartbeat(job_id)
        
        profile_key = "standardized_patient" if profile == "Medical Assessment" else "generic"
        
        def analysis_callback(message: str, percent: int):
            db.update_stage(job_id, 'analyzing_video', progress_details=message)
            db.update_heartbeat(job_id)
            print(f"   {message} ({percent}%)")
        
        analysis_result = flash_client.analyze_video(
            profile_type=profile_key,
            audio_transcript=audio_transcript if audio_transcript else None,
            progress_callback=analysis_callback
        )
        
        narrative = analysis_result.get("narrative", "")
        
        if narrative:
            db.save_narrative(job_id, narrative)
            
            narrative_file = save_analysis_file(
                narrative, 'narrative', video_filename, job_id
            )
            print(f"📄 Narrative saved: {narrative_file}")
        else:
            error_msg = "Flash 2.5 analysis produced no narrative"
            print(f"❌ {error_msg}")
            db.mark_error(job_id, error_msg)
            return
        
        # ========================================
        # Phase 5: Generate Assessment
        # ========================================
        print("\n" + "="*60)
        print("📋 Phase 5: Generating Assessment")
        print("="*60)
        
        db.update_stage(job_id, 'generating_assessment', 
                       progress_details="Generating medical faculty assessment")
        db.update_heartbeat(job_id)
        
        def assessment_callback(message: str, percent: int):
            db.update_stage(job_id, 'generating_assessment', progress_details=message)
            db.update_heartbeat(job_id)
            print(f"   {message} ({percent}%)")
        
        assessment = flash_client.generate_assessment(
            narrative=narrative,
            audio_transcript=audio_transcript if audio_transcript else None,
            progress_callback=assessment_callback
        )
        
        if assessment:
            db.save_assessment(job_id, assessment)
            
            assessment_file = save_analysis_file(
                assessment, 'assessment', video_filename, job_id
            )
            print(f"📄 Assessment saved: {assessment_file}")
        else:
            error_msg = "Flash 2.5 assessment generation failed"
            print(f"❌ {error_msg}")
            db.mark_error(job_id, error_msg)
            return
        
        # ========================================
        # Phase 6: Generate PDF
        # ========================================
        print("\n" + "="*60)
        print("📑 Phase 6: Generating PDF Report")
        print("="*60)
        
        db.update_stage(job_id, 'generating_pdf', 
                       progress_details="Creating PDF assessment report")
        db.update_heartbeat(job_id)
        
        output_dir = get_output_dir(video_filename, job_id)
        pdf_path = str(output_dir / "assessment_report.pdf")
        
        try:
            create_assessment_pdf(assessment, pdf_path)
            print(f"📄 PDF created: {pdf_path}")
            
            db.save_pdf_path(job_id, pdf_path)
            db.save_output_dir(job_id, str(output_dir))
            
            storage_manager = StorageManager()
            storage_key = storage_manager.upload_pdf(pdf_path, job_id, video_filename)
            
            if storage_key:
                db.save_storage_key(job_id, storage_key)
                print(f"☁️ PDF uploaded to object storage: {storage_key}")
            else:
                print("⚠️ Failed to upload PDF to object storage (continuing)")
                
        except Exception as pdf_error:
            error_logger.log_error("pdf_generation", job_id, video_filename,
                                  f"PDF generation failed: {str(pdf_error)}", pdf_error)
            print(f"⚠️ PDF generation failed: {pdf_error} (continuing without PDF)")
        
        # ========================================
        # Phase 7: Complete
        # ========================================
        print("\n" + "="*60)
        print("✅ Phase 7: Completing Job")
        print("="*60)
        
        db.update_stage(job_id, 'completed', status='completed',
                       progress_details="Analysis complete")
        
        error_logger.log_stage_exit("flash_processing", job_id, video_filename, success=True)
        
        print(f"\n🎉 Job {job_id} completed successfully!")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        error_msg = f"Flash processing failed: {str(e)}"
        error_logger.log_error("flash_processing", job_id, video_filename, error_msg, e)
        db.mark_error(job_id, error_msg)
        print(f"\n❌ Error: {error_msg}")
        traceback.print_exc()
        
    finally:
        # ========================================
        # Cleanup
        # ========================================
        print("\n" + "="*60)
        print("🧹 Cleanup")
        print("="*60)
        
        if flash_client:
            try:
                flash_client.cleanup()
            except Exception as cleanup_error:
                print(f"⚠️ Flash client cleanup error: {cleanup_error}")
        
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"🗑️ Removed temporary audio: {audio_path}")
            except Exception as e:
                print(f"⚠️ Could not remove audio file: {e}")
        
        if video_path and os.path.exists(video_path):
            if video_path.startswith('temp_videos/') or '/temp_' in video_path:
                try:
                    os.remove(video_path)
                    print(f"🗑️ Removed temporary video: {video_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove video file: {e}")
        
        print("✅ Cleanup complete")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_video_flash.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    process_video_flash(job_id)
