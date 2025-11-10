"""
AI Video Analysis - Clean Version
A video analysis application with scrollable output container.
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime

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
                    import tempfile
                    import json
                    from pathlib import Path
                    from video_processor import VideoProcessor, FrameBatchProcessor
                    from gpt5_client import GPT5Client
                    from profiles import ProfileManager
                    from utils import extract_audio_from_video, transcribe_audio_with_whisper
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from db_manager import AnalysisJobManager
                    
                    # Define processing phases
                    phases = [
                        "🎤 Phase 1: Audio Extraction",
                        "🗣️ Phase 2: Audio Transcription",
                        "🎬 Phase 3: Frame Extraction",
                        "🔍 Phase 4: Frame Analysis (AI)",
                        "✨ Phase 5: Narrative Synthesis (AI)",
                        "📊 Phase 6: Medical Assessment (AI)" if profile == "Medical Assessment" else None
                    ]
                    phases = [p for p in phases if p]  # Remove None entries
                    
                    # Create phase indicator container (using empty for in-place updates)
                    phase_container = st.empty()
                    
                    def update_phases(current_phase_index):
                        """Display all phases with current one highlighted."""
                        phase_display = "### 📍 Processing Phases\n\n"
                        for i, phase in enumerate(phases):
                            if i == current_phase_index:
                                phase_display += f"**➡️ {phase}** ⏳\n\n"
                            elif i < current_phase_index:
                                phase_display += f"~~{phase}~~ ✅\n\n"
                            else:
                                phase_display += f"{phase}\n\n"
                        phase_display += "---"
                        phase_container.markdown(phase_display)
                    
                    # Save uploaded video to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
                        tmp_video.write(uploaded_file.read())
                        video_path = tmp_video.name
                    
                    # Initialize database manager and create job
                    db = AnalysisJobManager()
                    job_id = db.create_job(
                        video_filename=uploaded_file.name,
                        video_path=video_path,
                        profile=profile,
                        fps=fps,
                        batch_size=batch_size
                    )
                    st.session_state.current_job_id = job_id
                    st.info(f"💾 Analysis job created: {job_id}")
                    
                    # Initialize variables that may be used in except block
                    audio_path = None
                    
                    try:
                        # Mark as in_progress INSIDE try block
                        db.update_stage(job_id, 'in_progress', status='in_progress')
                        
                        # Phase 1: Extract audio
                        update_phases(0)
                        audio_path = extract_audio_from_video(video_path)
                        
                        audio_transcript = ""
                        if audio_path:
                            # Phase 2: Transcribe audio
                            update_phases(1)
                            audio_container = st.empty()
                            # Transcribe with word-level timestamps for accurate faculty assessment
                            audio_transcript = transcribe_audio_with_whisper(
                                audio_path, 
                                api_key,
                                streamlit_container=audio_container
                            )
                            if not audio_transcript:
                                audio_container.warning("⚠️ No audio transcript available")
                            else:
                                # Save audio transcript only after successful transcription
                                db.save_audio_transcript(job_id, audio_transcript)
                                db.update_stage(job_id, 'audio_transcribed')
                                # Save transcript to file
                                transcript_file = save_analysis_file(
                                    audio_transcript, 'transcript', uploaded_file.name, job_id
                                )
                                st.info(f"📄 Transcript saved: {transcript_file}")
                        else:
                            st.warning("⚠️ No audio found in video (skipping transcription)")
                        
                        # Phase 3: Extract frames
                        update_phases(2)
                        processor = VideoProcessor()
                        processor.load_video(video_path, fps=fps)
                        
                        frames = processor.extract_frames()
                        st.success(f"✅ Extracted {len(frames)} frames at {fps} FPS")
                        
                        # Update stage after successful frame extraction
                        db.update_stage(job_id, 'frames_extracted')
                        
                        # Phase 4: Analyze frames
                        update_phases(3)
                        batch_processor = FrameBatchProcessor(batch_size=batch_size)
                        batches = batch_processor.create_batches(frames)
                        db.update_progress(job_id, 0, len(batches))
                        
                        # Get selected profile
                        profile_manager = ProfileManager()
                        profile_key = "standardized_patient" if profile == "Medical Assessment" else "generic"
                        selected_profile = profile_manager.get_profile(profile_key)
                        
                        # Debug: Verify profile type
                        if not isinstance(selected_profile, dict):
                            st.error(f"❌ Profile error: expected dict, got {type(selected_profile).__name__}")
                            raise TypeError(f"Profile must be dict, got {type(selected_profile).__name__}: {selected_profile}")
                        
                        # Initialize GPT-5 client
                        gpt5 = GPT5Client(api_key=api_key)
                        
                        # Process batches concurrently (30 at a time for Tier 4 limits)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        chunk_size = 30
                        total_batches = len(batches)
                        completed_batches = 0
                        
                        for chunk_start in range(0, total_batches, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, total_batches)
                            chunk_batches = batches[chunk_start:chunk_end]
                            
                            status_text.text(f"Analyzing batches {chunk_start+1}-{chunk_end} concurrently...")
                            
                            # Capture current context for this chunk
                            current_context = gpt5.context_state
                            
                            # Process all batches in this chunk concurrently
                            def process_batch(batch_data):
                                batch_idx, batch = batch_data
                                narrative, events = gpt5.analyze_frames(
                                    batch, 
                                    selected_profile, 
                                    current_context
                                )
                                return batch_idx, narrative, events
                            
                            batch_results = []
                            with ThreadPoolExecutor(max_workers=30) as executor:
                                futures = {
                                    executor.submit(process_batch, (chunk_start + i, batch)): i 
                                    for i, batch in enumerate(chunk_batches)
                                }
                                
                                try:
                                    # Timeout for each chunk (30 batches * 120s per batch = 3600s max)
                                    # Add some buffer for concurrency
                                    chunk_timeout = 3600  # 1 hour max per chunk
                                    for future in as_completed(futures, timeout=chunk_timeout):
                                        batch_idx, narrative, events = future.result(timeout=120)  # 2 min per batch result
                                        
                                        if narrative.startswith("Error during analysis"):
                                            st.error(f"❌ Analysis failed at batch {batch_idx+1}: {narrative}")
                                            raise Exception(narrative)
                                        
                                        batch_results.append((batch_idx, narrative, events))
                                        completed_batches += 1
                                        progress_bar.progress(completed_batches / total_batches)
                                        db.update_progress(job_id, completed_batches)
                                except TimeoutError:
                                    st.error(f"❌ Batch processing timed out. Some frames took too long to analyze.")
                                    raise Exception("Batch processing timeout - frames analysis took too long")
                            
                            # Sort results by batch index and update context
                            batch_results.sort(key=lambda x: x[0])
                            for _, narrative, events in batch_results:
                                gpt5.update_context(narrative, events)
                        
                        status_text.success("✅ Frame analysis complete!")
                        transcript = gpt5.get_full_transcript()
                        # Save frame transcript only after all batches complete successfully
                        db.save_frame_transcript(job_id, transcript)
                        db.update_stage(job_id, 'frames_analyzed')
                        
                        # Phase 5: Create enhanced narrative
                        update_phases(4)
                        events = gpt5.get_event_timeline()
                        
                        enhanced_narrative = gpt5.create_enhanced_narrative(
                            transcript=transcript,
                            events=events,
                            audio_transcript=audio_transcript,
                            profile=selected_profile
                        )
                        
                        # Check for errors to prevent cascading failures
                        if enhanced_narrative.startswith("Error during"):
                            st.error(f"❌ Narrative enhancement failed: {enhanced_narrative}")
                            raise Exception(enhanced_narrative)
                        
                        # Save narrative only after successful creation
                        db.save_narrative(job_id, enhanced_narrative)
                        db.update_stage(job_id, 'narrative_created')
                        # Save narrative to file
                        narrative_file = save_analysis_file(
                            enhanced_narrative, 'narrative', uploaded_file.name, job_id
                        )
                        st.info(f"📄 Narrative saved: {narrative_file}")
                        
                        # Phase 6: Generate medical assessment (if applicable)
                        assessment_report = ""
                        if profile == "Medical Assessment":
                            update_phases(5)
                            assessment_report = gpt5.assess_standardized_patient_encounter(enhanced_narrative)
                            
                            # Check for errors
                            if assessment_report.startswith("Error during"):
                                st.error(f"❌ Assessment generation failed: {assessment_report}")
                                raise Exception(assessment_report)
                            
                            # Save assessment only after successful creation
                            db.save_assessment(job_id, assessment_report)
                            # save_assessment already marks as completed with status='completed'
                            # Save assessment to file
                            assessment_file = save_analysis_file(
                                assessment_report, 'assessment', uploaded_file.name, job_id
                            )
                            st.info(f"📄 Assessment saved: {assessment_file}")
                        else:
                            # For non-medical profiles, mark as complete after narrative
                            db.update_stage(job_id, 'narrative_complete', status='completed')
                        
                        # Store results
                        st.session_state.transcript = transcript
                        st.session_state.enhanced_narrative = enhanced_narrative
                        st.session_state.assessment_report = assessment_report
                        st.session_state.audio_transcript = audio_transcript
                        st.session_state.analysis_complete = True
                        
                        # Cleanup temp files
                        if audio_path and Path(audio_path).exists():
                            Path(audio_path).unlink()
                            print(f"🗑️ Cleaned up audio file: {audio_path}")
                        if Path(video_path).exists():
                            Path(video_path).unlink()
                            print(f"🗑️ Cleaned up video file: {video_path}")
                        
                        st.success("🗑️ Temporary files cleaned up successfully")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        # Mark job as failed
                        db.mark_error(job_id, str(e))
                        # Cleanup temp files on error
                        if Path(video_path).exists():
                            Path(video_path).unlink()
                            print(f"🗑️ Cleaned up video file after error: {video_path}")
                        # Also cleanup audio file if it was created
                        if 'audio_path' in locals() and audio_path and Path(audio_path).exists():
                            Path(audio_path).unlink()
                            print(f"🗑️ Cleaned up audio file after error: {audio_path}")
            
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
                        st.write(st.session_state.assessment_report)
                        
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