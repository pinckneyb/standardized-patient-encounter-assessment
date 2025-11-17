"""
AI Video Analysis - Clean Version
A video analysis application with scrollable output container.
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from utils.error_logger import get_error_logger

def save_analysis_file(content: str, category: str, video_filename: str, job_id: str) -> str:
    """
    Save analysis results to organized folders.
    
    Args:
        content: The text content to save
        category: 'transcript', 'narrative', or 'assessment'
        video_filename: Original video filename
        job_id: Analysis job ID
        
    Returns:
        Path to saved file
    """
    # Create folder if it doesn't exist
    folder_map = {
        'transcript': 'transcripts',
        'narrative': 'narratives',
        'assessment': 'assessments'
    }
    
    folder = folder_map.get(category, 'output')
    Path(folder).mkdir(exist_ok=True)
    
    # Create filename with timestamp and job_id
    base_name = Path(video_filename).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_name}_{job_id}_{timestamp}.txt"
    filepath = Path(folder) / filename
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)

def display_assessment_report(assessment_text: str):
    """
    Display a beautifully formatted assessment report with color-coded scores.
    
    Args:
        assessment_text: JSON-formatted assessment report from AI
    """
    import json
    
    try:
        # Try to parse as JSON (handle markdown code blocks if present)
        clean_text = assessment_text.strip()
        if clean_text.startswith('```'):
            # Remove markdown code block markers
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_text
        
        assessment = json.loads(clean_text)
        
        # Define category order
        categories = [
            "History Taking",
            "Physical Examination", 
            "Communication Skills",
            "Clinical Reasoning",
            "Professionalism",
            "Patient Education and Closure"
        ]
        
        # Display scoring legend
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
            <b>Scoring Scale:</b> 
            <span style="color: #28a745; font-weight: bold;">5 = Excellent</span> | 
            <span style="color: #28a745;">4 = Above Average</span> | 
            <span style="color: #ffc107; font-weight: bold;">3 = Adequate</span> | 
            <span style="color: #dc3545;">2 = Below Average</span> | 
            <span style="color: #dc3545; font-weight: bold;">1 = Unsatisfactory</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display each category
        for category in categories:
            if category in assessment:
                data = assessment[category]
                
                # Handle both dict and string formats
                if isinstance(data, dict):
                    score = data.get('score', 'N/A')
                    feedback = data.get('feedback', 'No feedback provided.')
                else:
                    score = 'N/A'
                    feedback = str(data)
                
                # Determine color based on score
                try:
                    score_num = float(score) if score != 'N/A' else 0
                    if score_num >= 4:
                        color = "#28a745"  # Green
                        bg_color = "#d4edda"
                    elif score_num >= 3:
                        color = "#ffc107"  # Yellow
                        bg_color = "#fff3cd"
                    elif score_num > 0:
                        color = "#dc3545"  # Red
                        bg_color = "#f8d7da"
                    else:
                        color = "#6c757d"  # Grey
                        bg_color = "#e9ecef"
                except (ValueError, TypeError):
                    color = "#6c757d"
                    bg_color = "#e9ecef"
                
                # Display category with colored score
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #333;">{category}</h4>
                        <span style="background-color: {color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 16px;">
                            {score}/5
                        </span>
                    </div>
                    <p style="margin: 0; color: #555; line-height: 1.6;">{feedback}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display overall summary
        if "Overall" in assessment:
            overall = assessment["Overall"]
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 20px; border-radius: 8px; margin-top: 20px; border: 2px solid #2196f3;">
                <h4 style="margin-top: 0; color: #1976d2;">📋 Overall Assessment</h4>
                <p style="margin-bottom: 0; color: #333; line-height: 1.7; font-size: 15px;">{overall}</p>
            </div>
            """, unsafe_allow_html=True)
            
    except json.JSONDecodeError:
        # If not JSON, display as plain text with basic formatting
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff;">
            <pre style="white-space: pre-wrap; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; color: #333;">{assessment_text}</pre>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying assessment: {str(e)}")
        st.text(assessment_text)

# Page configuration
st.set_page_config(
    page_title="AI Video Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scrollable container
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .output-area {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">Standardized Patient Encounter Assessment</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'enhanced_narrative' not in st.session_state:
        st.session_state.enhanced_narrative = ""
    if 'assessment_report' not in st.session_state:
        st.session_state.assessment_report = ""
    if 'current_job_id' not in st.session_state:
        st.session_state.current_job_id = None
    
    # Clean up old incomplete jobs on startup
    try:
        from db_manager import AnalysisJobManager
        db_manager = AnalysisJobManager()
        db_manager.cleanup_old_incomplete_jobs()
    except Exception as e:
        pass
    
    # Job Status Polling UI
    st.markdown("---")
    with st.expander("🔄 Job Status Monitor", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("### Active Jobs")
        
        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        with col3:
            # Clear failed jobs button
            from db_manager import AnalysisJobManager
            db_temp = AnalysisJobManager()
            failed_count = len([j for j in db_temp.get_recent_jobs(limit=20) if j.get('status') == 'failed'])
            if failed_count > 0:
                if st.button(f"🗑️ Clear {failed_count} Failed"):
                    for job in db_temp.get_recent_jobs(limit=50):
                        if job.get('status') == 'failed':
                            db_temp.delete_job(job.get('job_id'))
                    st.success(f"Cleared {failed_count} failed jobs!")
                    st.rerun()
        
        try:
            from db_manager import AnalysisJobManager
            db_mgr = AnalysisJobManager()
            active_jobs = db_mgr.get_active_jobs()
            recent_jobs = db_mgr.get_recent_jobs(limit=5)
            
            if active_jobs:
                st.info("🔄 **Active jobs found!** Your analysis is running in the background.")
                
                for job in active_jobs:
                    job_id = job.get('job_id', 'Unknown')
                    video_filename = job.get('video_filename', 'Unknown')
                    status = job.get('status', 'unknown')
                    current_stage = job.get('current_stage', 'Unknown')
                    progress = job.get('progress', 0)
                    progress_details = job.get('progress_details', '')
                    created_at = job.get('created_at')
                    
                    st.markdown(f"#### Job: `{job_id}`")
                    st.markdown(f"**Video:** {video_filename}")
                    st.markdown(f"**Status:** {status.upper()}")
                    st.markdown(f"**Stage:** {current_stage}")
                    
                    if progress_details:
                        st.markdown(f"**Progress:** {progress_details}")
                    
                    if progress and progress > 0:
                        st.progress(progress / 100.0)
                        st.markdown(f"**{progress:.1f}% Complete**")
                    
                    if created_at:
                        elapsed_time = datetime.now() - created_at
                        minutes = int(elapsed_time.total_seconds() / 60)
                        seconds = int(elapsed_time.total_seconds() % 60)
                        st.markdown(f"**Time Elapsed:** {minutes}m {seconds}s")
                    
                    st.markdown("---")
                
                st.info("💡 **Tip:** Click 'Refresh Status' to see the latest progress. Processing will continue even if you close your browser.")
            
            elif recent_jobs:
                st.success("✅ No active jobs running")
                
                st.markdown("### Recent Jobs")
                for job in recent_jobs[:3]:
                    job_id = job.get('job_id', 'Unknown')
                    video_filename = job.get('video_filename', 'Unknown')
                    status = job.get('status', 'unknown')
                    created_at = job.get('created_at')
                    
                    status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏸️"
                    
                    with st.container():
                        col_job1, col_job2, col_job3, col_job4 = st.columns([2, 2, 1, 1])
                        with col_job1:
                            st.markdown(f"**{status_emoji} {video_filename}**")
                        with col_job2:
                            st.markdown(f"`{job_id}`")
                        with col_job3:
                            if status == "completed":
                                if st.button("View Results", key=f"view_{job_id}"):
                                    st.session_state.current_job_id = job_id
                                    st.rerun()
                        with col_job4:
                            if st.button("🗑️", key=f"delete_{job_id}", help="Delete this job"):
                                db_mgr.delete_job(job_id)
                                st.success(f"Deleted job {job_id}")
                                st.rerun()
            else:
                st.info("👋 No jobs found. Upload a video below to start your first analysis!")
            
            if st.session_state.current_job_id:
                st.markdown("---")
                st.markdown("### 📊 Job Results")
                
                job = db_mgr.get_job(st.session_state.current_job_id)
                if job:
                    status = job.get('status', 'unknown')
                    
                    if status == 'completed':
                        st.success(f"✅ Job `{st.session_state.current_job_id}` completed successfully!")
                        
                        audio_transcript = job.get('audio_transcript', '')
                        frame_transcript = job.get('frame_transcript', '')
                        enhanced_narrative = job.get('enhanced_narrative', '')
                        assessment_report = job.get('assessment_report', '')
                        video_filename = job.get('video_filename', 'result')
                        
                        if audio_transcript:
                            with st.expander("🎤 Audio Transcript", expanded=False):
                                st.text_area("Audio Transcript", audio_transcript, height=150, key="audio_result")
                        
                        if frame_transcript:
                            with st.expander("📝 Visual Analysis Transcript", expanded=False):
                                st.text_area("Frame Analysis", frame_transcript, height=150, key="frame_result")
                        
                        if enhanced_narrative:
                            with st.expander("✨ Enhanced Narrative", expanded=False):
                                st.write(enhanced_narrative)
                        
                        if assessment_report:
                            with st.expander("📊 Assessment Report", expanded=True):
                                display_assessment_report(assessment_report)
                                
                                from pdf_generator import create_assessment_pdf
                                import tempfile
                                
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                        pdf_path = create_assessment_pdf(
                                            assessment_report,
                                            tmp_pdf.name
                                        )
                                        
                                        with open(pdf_path, 'rb') as pdf_file:
                                            pdf_bytes = pdf_file.read()
                                        
                                        st.download_button(
                                            label="📥 Download Assessment PDF",
                                            data=pdf_bytes,
                                            file_name=f"assessment_report_{video_filename}.pdf",
                                            mime="application/pdf",
                                            key="pdf_download_result"
                                        )
                                        
                                        os.unlink(pdf_path)
                                        
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                        
                        if st.button("🗑️ Clear Results"):
                            st.session_state.current_job_id = None
                            st.rerun()
                    
                    elif status == 'failed':
                        st.error(f"❌ Job `{st.session_state.current_job_id}` failed!")
                        error_message = job.get('error_message', 'Unknown error')
                        st.error(f"**Error:** {error_message}")
                        
                        if st.button("🗑️ Clear Error"):
                            st.session_state.current_job_id = None
                            st.rerun()
                    
                    elif status in ('queued', 'in_progress'):
                        st.info(f"🔄 Job `{st.session_state.current_job_id}` is still processing. Click 'Refresh Status' to update.")
                else:
                    st.warning(f"Job `{st.session_state.current_job_id}` not found in database.")
                    if st.button("🗑️ Clear"):
                        st.session_state.current_job_id = None
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Error checking job status: {str(e)}")
    
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input - check environment variable first
        env_api_key = os.getenv('OPENAI_API_KEY', '')
        
        if env_api_key:
            st.success("🔑 API key loaded from secure storage")
            api_key = env_api_key
            # Show masked key for confirmation
            st.text_input(
                "OpenAI API Key",
                value="sk-" + "*" * 20,
                type="password",
                disabled=True,
                help="API key is securely stored and loaded automatically"
            )
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for video analysis"
            )
        
        # Profile selection
        profile = st.selectbox(
            "Analysis Profile",
            ["Medical Assessment", "Generic Video Narration"],
            help="Choose the type of analysis to perform"
        )
        
        # Processing settings
        st.subheader("Processing Settings")
        fps = st.slider("Frames per second", 0.5, 5.0, 1.0, 0.5)
        batch_size = st.slider("Batch size", 3, 15, 3)
        
        resolution_options = {
            "1080p (Full HD)": 1920,
            "720p (HD)": 1280,
            "480p (SD)": 854,
            "360p (Low)": 640
        }
        resolution_choice = st.selectbox(
            "Video Resolution",
            list(resolution_options.keys()),
            index=1,
            help="Lower resolutions process faster and use less memory"
        )
        max_resolution = resolution_options[resolution_choice]
        
    # Main content area
    st.header("📁 Video Input")
    
    # Video upload
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'm4v'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV, M4V (No size limit)"
    )
    
    # Create scrollable output container
    st.markdown("---")
    st.markdown("### 📋 Analysis Output")
    
    # Container for all output with scrollable styling
    with st.container():
        st.markdown('<div class="output-area">', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Video upload success
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Show video information
            video_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"📊 Video size: {video_size:.1f} MB")
            
            # API key check
            if not api_key:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start analysis")
            else:
                st.success("🔑 API key configured")
                
                # Analysis button
                if st.button("🚀 Start Analysis", type="primary"):
                    import subprocess
                    import sys
                    from pathlib import Path
                    from db_manager import AnalysisJobManager
                    
                    error_logger = get_error_logger()
                    
                    # Create temp_videos directory if it doesn't exist
                    temp_videos_dir = Path("temp_videos")
                    temp_videos_dir.mkdir(exist_ok=True)
                    
                    # Initialize database manager
                    db = AnalysisJobManager()
                    
                    # Generate job_id first
                    import uuid
                    job_id = str(uuid.uuid4())[:8]
                    
                    # Save uploaded video to persistent temp location with job_id in filename
                    video_filename = uploaded_file.name
                    video_suffix = Path(video_filename).suffix
                    persistent_video_path = temp_videos_dir / f"temp_{job_id}{video_suffix}"
                    
                    with open(persistent_video_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Create job in database - note: create_job returns its own job_id, but we need to use ours
                    # So we'll insert directly to ensure we use our generated job_id
                    try:
                        from db_manager import psycopg2
                        with db._get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO analysis_jobs 
                                    (job_id, video_filename, video_path, profile, fps, batch_size, 
                                     status, current_stage)
                                    VALUES (%s, %s, %s, %s, %s, %s, 'pending', 'created')
                                """, (job_id, video_filename, str(persistent_video_path), profile, fps, batch_size))
                                conn.commit()
                    except Exception as e:
                        error_logger.log_error("job_creation", job_id, video_filename, 
                                             f"Failed to create job: {str(e)}", e)
                        raise
                    
                    st.session_state.current_job_id = job_id
                    error_logger.log_info("job_creation", job_id, video_filename, 
                                         f"Job created: profile={profile}, fps={fps}, batch_size={batch_size}, size={video_size:.1f}MB")
                    
                    # Launch background processing subprocess
                    try:
                        process = subprocess.Popen(
                            [sys.executable, 'process_video_background.py', job_id],
                            start_new_session=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        
                        st.success(f"✅ Processing started in background (Job ID: {job_id})")
                        st.info("🔄 **You can safely close your browser.** Processing will continue independently.")
                        st.info("💡 Refresh this page later to check the status of your analysis.")
                        
                        error_logger.log_info("subprocess_launch", job_id, video_filename,
                                             f"Background subprocess launched with PID: {process.pid}")
                    except Exception as e:
                        error_msg = f"Failed to launch background process: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        error_logger.log_error("subprocess_launch", job_id, video_filename, error_msg, e)
                        db.mark_error(job_id, error_msg)
                        
                        # Clean up video file if subprocess launch failed
                        if persistent_video_path.exists():
                            persistent_video_path.unlink()
                    
                    # Don't process inline - that's now handled by background worker
                        
            
            # Display analysis results
            if st.session_state.analysis_complete:
                st.success("✅ Analysis Complete!")
                
                # Audio Transcript
                if 'audio_transcript' in st.session_state and st.session_state.audio_transcript:
                    with st.expander("🎤 Audio Transcript (Streaming)", expanded=True):
                        st.text_area("Audio Transcript", st.session_state.audio_transcript, height=150)
                
                # Visual Analysis Transcript
                if st.session_state.transcript:
                    with st.expander("📝 Visual Analysis Transcript", expanded=True):
                        st.text_area("Frame Analysis", st.session_state.transcript, height=150)
                
                # Enhanced Narrative
                if st.session_state.enhanced_narrative:
                    with st.expander("✨ Enhanced Narrative", expanded=True):
                        st.write(st.session_state.enhanced_narrative)
                
                # Assessment Report
                if st.session_state.assessment_report:
                    with st.expander("📊 Assessment Report", expanded=True):
                        display_assessment_report(st.session_state.assessment_report)
                        
                        # PDF download button
                        from pdf_generator import create_assessment_pdf
                        import tempfile
                        
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                pdf_path = create_assessment_pdf(
                                    st.session_state.assessment_report,
                                    tmp_pdf.name
                                )
                                
                                with open(pdf_path, 'rb') as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                
                                st.download_button(
                                    label="📥 Download Assessment PDF",
                                    data=pdf_bytes,
                                    file_name=f"assessment_report_{uploaded_file.name}.pdf",
                                    mime="application/pdf"
                                )
                                
                                os.unlink(pdf_path)
                                
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                
                # Download options
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Download Transcript"):
                        st.download_button(
                            label="📥 Download Transcript",
                            data=st.session_state.transcript,
                            file_name=f"transcript_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    if st.button("✨ Download Enhanced Narrative"):
                        st.download_button(
                            label="📥 Download Narrative",
                            data=st.session_state.enhanced_narrative,
                            file_name=f"enhanced_narrative_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
        
        elif uploaded_file is None:
            st.info("👆 Please upload a video file to begin analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()