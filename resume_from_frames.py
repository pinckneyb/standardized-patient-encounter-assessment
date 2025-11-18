#!/usr/bin/env python3
"""
Resume analysis from saved frame transcript.
Usage: python resume_from_frames.py <job_id>
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import AnalysisJobManager
from gpt5_client import GPT5Client
from profiles import ProfileManager
from utils.error_logger import ErrorLogger
from datetime import datetime

# Initialize error logger
error_logger = ErrorLogger()

def save_analysis_file(content: str, category: str, video_filename: str, job_id: str) -> str:
    """Save analysis results to organized folders."""
    folder_map = {
        'transcript': 'transcripts',
        'narrative': 'narratives',
        'assessment': 'assessments'
    }
    
    folder = folder_map.get(category, 'output')
    Path(folder).mkdir(exist_ok=True)
    
    base_name = Path(video_filename).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_name}_{job_id}_{timestamp}.txt"
    filepath = Path(folder) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)

def resume_analysis(job_id):
    """Resume analysis from saved frame transcript"""
    print(f"📂 Resuming analysis for job: {job_id}")
    
    # Initialize database
    db = AnalysisJobManager()
    
    # Get job details
    job = db.get_job(job_id)
    if not job:
        print(f"❌ Job {job_id} not found")
        return False
    
    video_filename = job['video_filename']
    profile = job['profile']
    
    print(f"Video: {video_filename}")
    print(f"Profile: {profile}")
    
    # Get saved transcripts
    frame_transcript = job['frame_transcript']
    audio_transcript = job['audio_transcript']
    
    if not frame_transcript:
        print("❌ No frame transcript found. Cannot resume.")
        return False
    
    print(f"✅ Found frame transcript: {len(frame_transcript)} characters")
    print(f"✅ Found audio transcript: {len(audio_transcript) if audio_transcript else 0} characters")
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    # Get profile
    profile_manager = ProfileManager()
    profile_key = "standardized_patient" if profile == "Medical Assessment" else "generic"
    selected_profile = profile_manager.get_profile(profile_key)
    
    # Initialize GPT client
    gpt5 = GPT5Client(api_key=api_key)
    gpt5.set_job_context(job_id, video_filename)
    
    # Parse the existing frame transcript to reconstruct event timeline
    print("📋 Reconstructing event timeline from transcript...")
    gpt5.context_state = frame_transcript
    events = gpt5.get_event_timeline()
    print(f"✅ Reconstructed {len(events)} events")
    
    try:
        # Phase 5: Narrative Synthesis
        print("✨ Phase 5: Narrative Synthesis (AI)")
        error_logger.log_stage_entry("narrative_synthesis", job_id, video_filename)
        db.update_stage(job_id, 'creating_narrative', progress_details="Synthesizing narrative from analysis")
        
        enhanced_narrative = gpt5.create_enhanced_narrative(
            transcript=frame_transcript,
            events=events,
            audio_transcript=audio_transcript or "",
            profile=selected_profile
        )
        
        if enhanced_narrative.startswith("Error during"):
            error_logger.log_error("narrative_synthesis", job_id, video_filename,
                                 f"Narrative enhancement failed: {enhanced_narrative}", None)
            db.mark_error(job_id, f"Narrative enhancement failed: {enhanced_narrative}")
            print(f"❌ Narrative enhancement failed: {enhanced_narrative}")
            return False
        
        db.save_narrative(job_id, enhanced_narrative)
        db.update_stage(job_id, 'narrative_created', progress_details="Enhanced narrative created successfully")
        narrative_file = save_analysis_file(enhanced_narrative, 'narrative', video_filename, job_id)
        print(f"📄 Narrative saved: {narrative_file}")
        error_logger.log_stage_exit("narrative_synthesis", job_id, video_filename, success=True)
        
        # Phase 6: Medical Assessment (if Medical profile)
        if profile == "Medical Assessment":
            print("🏥 Phase 6: Medical Assessment (AI)")
            error_logger.log_stage_entry("medical_assessment", job_id, video_filename)
            db.update_stage(job_id, 'generating_assessment', progress_details="Generating medical assessment report")
            
            assessment_report = gpt5.assess_standardized_patient_encounter(
                narrative=enhanced_narrative
            )
            
            if assessment_report.startswith("Error"):
                error_logger.log_error("medical_assessment", job_id, video_filename,
                                     f"Assessment generation failed: {assessment_report}", None)
                db.mark_error(job_id, f"Assessment generation failed: {assessment_report}")
                print(f"❌ Assessment generation failed: {assessment_report}")
                return False
            
            db.save_assessment(job_id, assessment_report)
            db.update_stage(job_id, 'assessment_generated', progress_details="Medical assessment report generated")
            assessment_file = save_analysis_file(assessment_report, 'assessment', video_filename, job_id)
            print(f"📄 Assessment saved: {assessment_file}")
            error_logger.log_stage_exit("medical_assessment", job_id, video_filename, success=True)
        
        # Mark complete
        db.mark_complete(job_id)
        print("✅ Analysis completed successfully!")
        print(f"📊 Results available for job: {job_id}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Resume failed: {error_msg}")
        import traceback
        print(traceback.format_exc())
        error_logger.log_error("resume_processing", job_id, video_filename,
                             f"Resume failed with error: {error_msg}", e)
        db.mark_error(job_id, error_msg)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resume_from_frames.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    success = resume_analysis(job_id)
    sys.exit(0 if success else 1)
