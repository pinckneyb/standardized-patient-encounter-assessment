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
                    import tempfile
                    import json
                    from pathlib import Path
                    from video_processor import VideoProcessor, FrameBatchProcessor
                    from gpt5_client import GPT5Client
                    from profiles import ProfileManager
                    from audio_utils import extract_audio_from_video, transcribe_audio_with_whisper
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from db_manager import AnalysisJobManager
                    
                    error_logger = get_error_logger()
                    st.warning("⏱️ **Important:** Large videos may take 15-30 minutes to process. Please keep your browser open during analysis.")
                    
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
                    error_logger.log_info("job_creation", job_id, uploaded_file.name, 
                                         f"Job created: profile={profile}, fps={fps}, batch_size={batch_size}, size={video_size:.1f}MB")
                    
                    # Initialize variables that may be used in except block
                    audio_path = None
                    
                    try:
                        # Mark as in_progress INSIDE try block
                        db.update_stage(job_id, 'in_progress', status='in_progress',
                                       progress_details="Starting video analysis")
                        error_logger.log_stage_entry("processing", job_id, uploaded_file.name)
                        
                        # Phase 1: Extract audio
                        update_phases(0)
                        error_logger.log_stage_entry("audio_extraction", job_id, uploaded_file.name)
                        db.update_stage(job_id, 'extracting_audio', progress_details="Extracting audio track from video")
                        audio_path = extract_audio_from_video(video_path)
                        error_logger.log_stage_exit("audio_extraction", job_id, uploaded_file.name, success=True)
                        
                        audio_transcript = ""
                        if audio_path:
                            # Phase 2: Transcribe audio
                            update_phases(1)
                            error_logger.log_stage_entry("audio_transcription", job_id, uploaded_file.name)
                            db.update_stage(job_id, 'transcribing_audio', progress_details="Transcribing audio using Whisper API")
                            audio_container = st.empty()
                            audio_transcript = transcribe_audio_with_whisper(
                                audio_path, 
                                api_key,
                                streamlit_container=audio_container
                            )
                            if not audio_transcript:
                                audio_container.warning("⚠️ No audio transcript available")
                                error_logger.log_warning("audio_transcription", job_id, uploaded_file.name,
                                                        "No audio transcript generated")
                            else:
                                db.save_audio_transcript(job_id, audio_transcript)
                                db.update_stage(job_id, 'audio_transcribed', progress_details="Audio transcription completed")
                                transcript_file = save_analysis_file(
                                    audio_transcript, 'transcript', uploaded_file.name, job_id
                                )
                                st.info(f"📄 Transcript saved: {transcript_file}")
                                error_logger.log_stage_exit("audio_transcription", job_id, uploaded_file.name, success=True)
                        else:
                            st.warning("⚠️ No audio found in video (skipping transcription)")
                            error_logger.log_warning("audio_extraction", job_id, uploaded_file.name,
                                                    "No audio found in video")
                        
                        # Phase 3: Prepare streaming frame extraction
                        update_phases(2)
                        error_logger.log_stage_entry("frame_extraction", job_id, uploaded_file.name)
                        db.update_stage(job_id, 'loading_video', progress_details=f"Loading video for frame extraction at {fps} FPS")
                        processor = VideoProcessor()
                        processor.set_job_context(job_id, uploaded_file.name)
                        processor.load_video(video_path, fps=fps)
                        
                        # Calculate estimated total batches for progress tracking
                        # Use duration * fps to get actual sampled frames (not total raw frames)
                        import math
                        if processor.duration and processor.duration > 0:
                            estimated_sampled_frames = processor.duration * fps
                            estimated_total_batches = math.ceil(estimated_sampled_frames / batch_size)
                        else:
                            estimated_total_batches = None  # Unknown, will track incrementally
                        
                        st.info(f"📊 Preparing to extract frames at {fps} FPS (estimated {estimated_total_batches} batches)" if estimated_total_batches else f"📊 Preparing to extract frames at {fps} FPS")
                        
                        # Initialize progress tracking
                        if estimated_total_batches:
                            db.update_progress(job_id, 0, estimated_total_batches)
                        
                        # Phase 4: Analyze frames using streaming batches
                        update_phases(3)
                        batch_processor = FrameBatchProcessor(batch_size=batch_size)
                        
                        # Create streaming frame iterator
                        frame_iterator = processor.iter_frames_streaming(max_resolution=max_resolution)
                        
                        # Track if any frames were extracted
                        frames_extracted_flag = False
                        total_frames_processed = 0
                        
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
                        gpt5.set_job_context(job_id, uploaded_file.name)
                        error_logger.log_info("gpt5_initialization", job_id, uploaded_file.name,
                                             "GPT-5 client initialized successfully")
                        
                        # Process batches using streaming chunks (30 at a time for Tier 4 limits)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        chunk_size = 10
                        completed_batches = 0
                        chunk_index = 0
                        
                        error_logger.log_stage_entry("frame_analysis", job_id, uploaded_file.name)
                        
                        # Process streaming chunked batches
                        for chunk_batches in batch_processor.iter_chunked_batches(frame_iterator, chunk_size=chunk_size):
                            chunk_index += 1
                            chunk_start = completed_batches
                            chunk_end = chunk_start + len(chunk_batches)
                            
                            # Mark frames_extracted after first batch
                            if not frames_extracted_flag and len(chunk_batches) > 0:
                                db.update_stage(job_id, 'frames_extracted')
                                frames_extracted_flag = True
                                st.success(f"✅ Started extracting frames (streaming)")
                            
                            status_text.text(f"Analyzing batches {chunk_start+1}-{chunk_end} concurrently (chunk {chunk_index})...")
                            
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
                            with ThreadPoolExecutor(max_workers=10) as executor:
                                futures = {
                                    executor.submit(process_batch, (chunk_start + i, batch)): i 
                                    for i, batch in enumerate(chunk_batches)
                                }
                                
                                try:
                                    # Timeout for each chunk (10 batches * 120s per batch = 1200s max)
                                    chunk_timeout = 1200  # 20 minutes max per chunk
                                    for future in as_completed(futures, timeout=chunk_timeout):
                                        batch_idx, narrative, events = future.result(timeout=120)  # 2 min per batch result
                                        
                                        if narrative.startswith("Error during analysis"):
                                            error_logger.log_error("batch_processing", job_id, uploaded_file.name,
                                                                 f"Analysis failed at batch {batch_idx+1}: {narrative}", None)
                                            db.mark_error(job_id, f"Batch {batch_idx+1} analysis failed: {narrative}")
                                            st.error(f"❌ Analysis failed at batch {batch_idx+1}: {narrative}")
                                            raise Exception(narrative)
                                        
                                        batch_results.append((batch_idx, narrative, events))
                                        completed_batches += 1
                                        
                                        # Explicit cleanup: release frame data from memory
                                        batch_frames = chunk_batches[futures[future]]
                                        total_frames_processed += len(batch_frames)
                                        for frame_data in batch_frames:
                                            if 'frame' in frame_data:
                                                del frame_data['frame']
                                            if 'frame_pil' in frame_data:
                                                frame_data['frame_pil'].close()
                                                del frame_data['frame_pil']
                                        
                                        # Update progress with detailed logging
                                        if estimated_total_batches:
                                            progress_bar.progress(min(1.0, completed_batches / estimated_total_batches))
                                            db.update_progress(job_id, completed_batches, estimated_total_batches)
                                            db.update_stage(job_id, 'analyzing_frames',
                                                          progress_details=f"Analyzing batch {completed_batches}/{estimated_total_batches}")
                                            if completed_batches % 5 == 0:
                                                error_logger.log_progress("frame_analysis", job_id, uploaded_file.name,
                                                                        completed_batches, estimated_total_batches,
                                                                        f"Frames processed: {total_frames_processed}")
                                        else:
                                            db.update_progress(job_id, completed_batches, completed_batches)
                                            db.update_stage(job_id, 'analyzing_frames',
                                                          progress_details=f"Analyzing batch {completed_batches}")
                                except TimeoutError as timeout_error:
                                    error_logger.log_error("batch_processing", job_id, uploaded_file.name,
                                                         "Batch processing timeout", timeout_error)
                                    db.mark_error(job_id, "Batch processing timeout - frames analysis took too long")
                                    st.error(f"❌ Batch processing timed out. Some frames took too long to analyze.")
                                    raise Exception("Batch processing timeout - frames analysis took too long")
                            
                            # Sort results by batch index and update context
                            batch_results.sort(key=lambda x: x[0])
                            for _, narrative, events in batch_results:
                                gpt5.update_context(narrative, events)
                            
                            # Cleanup chunk batches and trigger garbage collection
                            del chunk_batches
                            del batch_results
                            import gc
                            gc.collect()
                        
                        # If no frames were extracted, warn but don't fail
                        if not frames_extracted_flag:
                            st.warning("⚠️ No frames were extracted from video")
                            db.update_stage(job_id, 'frames_extracted')
                        
                        status_text.success("✅ Frame analysis complete!")
                        error_logger.log_stage_exit("frame_analysis", job_id, uploaded_file.name, success=True)
                        transcript = gpt5.get_full_transcript()
                        db.save_frame_transcript(job_id, transcript)
                        db.update_stage(job_id, 'frames_analyzed', progress_details="All frames analyzed successfully")
                        
                        # Phase 5: Create enhanced narrative
                        update_phases(4)
                        error_logger.log_stage_entry("narrative_synthesis", job_id, uploaded_file.name)
                        db.update_stage(job_id, 'creating_narrative', progress_details="Synthesizing narrative from analysis")
                        events = gpt5.get_event_timeline()
                        
                        enhanced_narrative = gpt5.create_enhanced_narrative(
                            transcript=transcript,
                            events=events,
                            audio_transcript=audio_transcript,
                            profile=selected_profile
                        )
                        
                        if enhanced_narrative.startswith("Error during"):
                            error_logger.log_error("narrative_synthesis", job_id, uploaded_file.name,
                                                 f"Narrative enhancement failed: {enhanced_narrative}", None)
                            db.mark_error(job_id, f"Narrative enhancement failed: {enhanced_narrative}")
                            st.error(f"❌ Narrative enhancement failed: {enhanced_narrative}")
                            raise Exception(enhanced_narrative)
                        
                        db.save_narrative(job_id, enhanced_narrative)
                        db.update_stage(job_id, 'narrative_created', progress_details="Enhanced narrative created successfully")
                        narrative_file = save_analysis_file(
                            enhanced_narrative, 'narrative', uploaded_file.name, job_id
                        )
                        st.info(f"📄 Narrative saved: {narrative_file}")
                        error_logger.log_stage_exit("narrative_synthesis", job_id, uploaded_file.name, success=True)
                        
                        # Phase 6: Generate medical assessment (if applicable)
                        assessment_report = ""
                        if profile == "Medical Assessment":
                            update_phases(5)
                            error_logger.log_stage_entry("assessment_generation", job_id, uploaded_file.name)
                            db.update_stage(job_id, 'generating_assessment', 
                                          progress_details="Generating medical assessment report")
                            assessment_report = gpt5.assess_standardized_patient_encounter(enhanced_narrative)
                            
                            if assessment_report.startswith("Error during"):
                                error_logger.log_error("assessment_generation", job_id, uploaded_file.name,
                                                     f"Assessment generation failed: {assessment_report}", None)
                                db.mark_error(job_id, f"Assessment generation failed: {assessment_report}")
                                st.error(f"❌ Assessment generation failed: {assessment_report}")
                                raise Exception(assessment_report)
                            
                            db.save_assessment(job_id, assessment_report)
                            assessment_file = save_analysis_file(
                                assessment_report, 'assessment', uploaded_file.name, job_id
                            )
                            st.info(f"📄 Assessment saved: {assessment_file}")
                            error_logger.log_stage_exit("assessment_generation", job_id, uploaded_file.name, success=True)
                        else:
                            db.update_stage(job_id, 'narrative_complete', status='completed',
                                          progress_details="Analysis completed successfully")
                        
                        # Store results
                        st.session_state.transcript = transcript
                        st.session_state.enhanced_narrative = enhanced_narrative
                        st.session_state.assessment_report = assessment_report
                        st.session_state.audio_transcript = audio_transcript
                        st.session_state.analysis_complete = True
                        
                        # Cleanup temp files
                        if audio_path and Path(audio_path).exists():
                            Path(audio_path).unlink()
                            error_logger.log_info("cleanup", job_id, uploaded_file.name,
                                                 f"Cleaned up audio file: {audio_path}")
                        if Path(video_path).exists():
                            Path(video_path).unlink()
                            error_logger.log_info("cleanup", job_id, uploaded_file.name,
                                                 f"Cleaned up video file: {video_path}")
                        
                        error_logger.log_stage_exit("processing", job_id, uploaded_file.name, success=True)
                        error_logger.log_info("job_completion", job_id, uploaded_file.name,
                                             f"Analysis completed successfully - Total batches: {completed_batches}, Total frames: {total_frames_processed}")
                        st.success("🗑️ Temporary files cleaned up successfully")
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Analysis failed: {error_msg}")
                        error_logger.log_error("processing", job_id, uploaded_file.name,
                                             f"Analysis failed with error: {error_msg}", e)
                        error_logger.log_stage_exit("processing", job_id, uploaded_file.name, success=False)
                        
                        db.mark_error(job_id, error_msg)
                        
                        if Path(video_path).exists():
                            Path(video_path).unlink()
                            error_logger.log_info("cleanup", job_id, uploaded_file.name,
                                                 f"Cleaned up video file after error: {video_path}")
                        
                        if 'audio_path' in locals() and audio_path and Path(audio_path).exists():
                            Path(audio_path).unlink()
                            error_logger.log_info("cleanup", job_id, uploaded_file.name,
                                                 f"Cleaned up audio file after error: {audio_path}")
            
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