#!/usr/bin/env python3
"""
Background video processing worker - runs independently of Streamlit session.
This allows processing to continue even if the browser disconnects.

Usage: python process_video_background.py <job_id>
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import traceback
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


def get_output_dir(video_filename: str, job_id: str) -> Path:
    """
    Get the output directory for a specific video analysis job.
    Creates the directory if it doesn't exist.
    
    Args:
        video_filename: Original video filename
        job_id: Analysis job ID
        
    Returns:
        Path object for the job's output directory
    """
    base_name = Path(video_filename).stem
    output_dir = Path('outputs') / f"{base_name}_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_analysis_file(content: str, category: str, video_filename: str, job_id: str) -> str:
    """
    Save analysis results to per-job output folder.
    
    Args:
        content: The text content to save
        category: 'transcript', 'narrative', or 'assessment'
        video_filename: Original video filename
        job_id: Analysis job ID
        
    Returns:
        Path to saved file
    """
    # Get job-specific output directory
    output_dir = get_output_dir(video_filename, job_id)
    
    # Use standardized filenames
    filename_map = {
        'transcript': 'transcript.txt',
        'narrative': 'narrative.txt',
        'assessment': 'assessment.txt'
    }
    
    filename = filename_map.get(category, f'{category}.txt')
    filepath = output_dir / filename
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)


def run_stage_with_timeout(stage_name: str, timeout_seconds: int, callable_func: Callable, 
                           db, job_id: str, error_logger, video_filename: str):
    """
    Execute a processing stage with timeout protection, heartbeat updates, and error handling.
    
    This wrapper ensures:
    1. Heartbeat updates every 30 seconds to prove process is alive
    2. Timeout enforcement to prevent infinite hangs
    3. Proper error logging and database updates on failure
    4. Clean recovery even if API calls stall
    
    NOTE: This wrapper does NOT update the stage in the database. The callable itself
    is responsible for its own stage transitions. This prevents duplicate stage updates.
    
    Args:
        stage_name: Name of the stage (for logging only, not DB updates)
        timeout_seconds: Maximum execution time before forcing timeout
        callable_func: The function to execute (must return result or raise exception)
        db: Database manager instance
        job_id: Job ID for tracking
        error_logger: Error logger instance
        video_filename: Video filename for logging
        
    Returns:
        Result from callable_func
        
    Raises:
        Exception: If stage times out or callable_func fails
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    # Initialize heartbeat (but don't update stage - let callable handle it)
    db.update_heartbeat(job_id)
    error_logger.log_stage_entry(stage_name, job_id, video_filename)
    print(f"\n{'='*60}")
    print(f"⏱️  Starting stage: {stage_name} (timeout: {timeout_seconds}s)")
    print(f"{'='*60}")
    
    # Heartbeat thread to update DB every 30 seconds
    stop_heartbeat = threading.Event()
    
    def heartbeat_worker():
        while not stop_heartbeat.is_set():
            stop_heartbeat.wait(30)  # Update every 30 seconds
            if not stop_heartbeat.is_set():
                try:
                    db.update_heartbeat(job_id)
                    print(f"💓 Heartbeat: {stage_name} still running...")
                except Exception as e:
                    error_logger.log_error(stage_name, job_id, video_filename,
                                         f"Heartbeat update failed: {str(e)}", e)
    
    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()
    
    try:
        # Execute callable with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callable_func)
            try:
                result = future.result(timeout=timeout_seconds)
                print(f"✅ Stage {stage_name} completed successfully")
                error_logger.log_stage_exit(stage_name, job_id, video_filename, success=True)
                return result
                
            except FutureTimeoutError:
                # Timeout - cancel the future and mark as failed
                future.cancel()
                error_msg = f"Stage '{stage_name}' timed out after {timeout_seconds} seconds"
                error_logger.log_error(stage_name, job_id, video_filename, error_msg, None)
                db.mark_error(job_id, error_msg)
                print(f"❌ TIMEOUT: {error_msg}")
                raise Exception(error_msg)
                
            except Exception as e:
                # Callable raised an exception
                error_msg = f"Stage '{stage_name}' failed: {str(e)}"
                error_logger.log_error(stage_name, job_id, video_filename, error_msg, e)
                db.mark_error(job_id, error_msg)
                print(f"❌ ERROR: {error_msg}")
                raise
                
    finally:
        # Stop heartbeat thread
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)


def process_video_job(job_id: str):
    """
    Process a video analysis job in the background.
    
    Args:
        job_id: The analysis job ID to process
    """
    from db_manager import AnalysisJobManager
    from video_processor import VideoProcessor, FrameBatchProcessor
    from gpt5_client import GPT5Client
    from profiles import ProfileManager
    from audio_utils import extract_audio_from_video, transcribe_audio_with_whisper
    from utils.error_logger import get_error_logger
    
    error_logger = get_error_logger()
    db = AnalysisJobManager()
    
    # Load job details
    job = db.get_job(job_id)
    if not job:
        print(f"❌ Job {job_id} not found in database")
        return
    
    video_filename = job['video_filename']
    video_path = job['video_path']
    profile = job['profile']
    fps = job['fps']
    batch_size = job['batch_size']
    
    print(f"🚀 Starting background processing for job {job_id}")
    print(f"   Video: {video_filename}")
    print(f"   Profile: {profile}")
    print(f"   Settings: fps={fps}, batch_size={batch_size}")
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        error_msg = "OPENAI_API_KEY not found in environment"
        print(f"❌ {error_msg}")
        db.mark_error(job_id, error_msg)
        return
    
    audio_path = None
    
    try:
        # Mark as in_progress
        db.update_stage(job_id, 'in_progress', status='in_progress',
                       progress_details="Starting video analysis")
        error_logger.log_stage_entry("processing", job_id, video_filename)
        
        # Phase 1: Extract audio
        print("🎤 Phase 1: Audio Extraction")
        error_logger.log_stage_entry("audio_extraction", job_id, video_filename)
        db.update_stage(job_id, 'extracting_audio', progress_details="Extracting audio track from video")
        audio_path = extract_audio_from_video(video_path)
        error_logger.log_stage_exit("audio_extraction", job_id, video_filename, success=True)
        
        audio_transcript = ""
        if audio_path:
            # Phase 2: Transcribe audio
            # Note: Whisper API has built-in 6-minute timeout, so no need for wrapper
            print("🗣️ Phase 2: Audio Transcription")
            error_logger.log_stage_entry("audio_transcription", job_id, video_filename)
            
            try:
                db.update_stage(job_id, 'transcribing_audio', progress_details="Transcribing audio using Whisper API")
                db.update_heartbeat(job_id)  # Initial heartbeat
                
                audio_transcript = transcribe_audio_with_whisper(audio_path, api_key)
                
                if not audio_transcript:
                    print("⚠️ No audio transcript available")
                    error_logger.log_warning("audio_transcription", job_id, video_filename,
                                            "No audio transcript generated")
                else:
                    db.save_audio_transcript(job_id, audio_transcript)
                    db.update_stage(job_id, 'audio_transcribed', progress_details="Audio transcription completed")
                    db.update_heartbeat(job_id)  # Update after completion
                    
                    # Save transcript file
                    transcript_file = save_analysis_file(
                        audio_transcript, 'transcript', video_filename, job_id
                    )
                    print(f"📄 Transcript saved: {transcript_file}")
                    error_logger.log_stage_exit("audio_transcription", job_id, video_filename, success=True)
            
            except Exception as e:
                error_msg = f"Transcription failed: {str(e)}"
                error_logger.log_error("audio_transcription", job_id, video_filename, error_msg, e)
                db.mark_error(job_id, error_msg)
                raise
        else:
            print("⚠️ No audio found in video (skipping transcription)")
            error_logger.log_warning("audio_extraction", job_id, video_filename,
                                    "No audio found in video")
        
        # Phase 3: Prepare streaming frame extraction
        print("🎬 Phase 3: Frame Extraction")
        error_logger.log_stage_entry("frame_extraction", job_id, video_filename)
        db.update_stage(job_id, 'loading_video', progress_details=f"Loading video for frame extraction at {fps} FPS")
        processor = VideoProcessor()
        processor.set_job_context(job_id, video_filename)
        processor.load_video(video_path, fps=fps)
        
        # Calculate estimated total batches
        # CRITICAL FIX: Use floor() instead of ceil() to avoid over-estimating
        # Over-estimation causes the frame iterator to wait forever for non-existent frames
        if processor.duration and processor.duration > 0:
            estimated_sampled_frames = processor.duration * fps
            estimated_total_batches = int(estimated_sampled_frames / batch_size)  # floor, not ceil
        else:
            estimated_total_batches = None
        
        print(f"📊 Preparing to extract frames at {fps} FPS (estimated {estimated_total_batches} batches)" if estimated_total_batches else f"📊 Preparing to extract frames at {fps} FPS")
        
        # Initialize progress tracking
        if estimated_total_batches:
            db.update_progress(job_id, 0, estimated_total_batches)
        
        # Phase 4: Analyze frames using streaming batches
        print("🔍 Phase 4: Frame Analysis (AI)")
        batch_processor = FrameBatchProcessor(batch_size=batch_size)
        
        # Create streaming frame iterator - use default 720p resolution
        max_resolution = 1280
        frame_iterator = processor.iter_frames_streaming(max_resolution=max_resolution)
        
        # Track if any frames were extracted
        frames_extracted_flag = False
        total_frames_processed = 0
        
        # Get selected profile
        profile_manager = ProfileManager()
        profile_key = "standardized_patient" if profile == "Medical Assessment" else "generic"
        selected_profile = profile_manager.get_profile(profile_key)
        
        # Verify profile type
        if not isinstance(selected_profile, dict):
            raise TypeError(f"Profile must be dict, got {type(selected_profile).__name__}: {selected_profile}")
        
        # Initialize GPT-5 client
        gpt5 = GPT5Client(api_key=api_key)
        gpt5.set_job_context(job_id, video_filename)
        error_logger.log_info("gpt5_initialization", job_id, video_filename,
                             "GPT-5 client initialized successfully")
        
        # Process batches using streaming chunks (30 at a time for API limits)
        chunk_size = 30
        completed_batches = 0
        chunk_index = 0
        
        error_logger.log_stage_entry("frame_analysis", job_id, video_filename)
        
        # Process streaming chunked batches
        for chunk_batches in batch_processor.iter_chunked_batches(frame_iterator, chunk_size=chunk_size):
            chunk_index += 1
            chunk_start = completed_batches
            chunk_end = chunk_start + len(chunk_batches)
            
            # CRITICAL: Check if we've already processed all expected batches
            if estimated_total_batches and completed_batches >= estimated_total_batches:
                print(f"✅ All {estimated_total_batches} batches completed, stopping iteration")
                break
            
            # Mark frames_extracted after first batch
            if not frames_extracted_flag and len(chunk_batches) > 0:
                db.update_stage(job_id, 'frames_extracted')
                frames_extracted_flag = True
                print(f"✅ Started extracting frames (streaming)")
            
            print(f"Analyzing batches {chunk_start+1}-{chunk_end} concurrently (chunk {chunk_index})...")
            
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
            num_batches = len(chunk_batches)
            
            # Memory monitoring setup
            import psutil
            import gc
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Memory before chunk {chunk_index}: {mem_before:.1f} MB RSS")
            
            # CRITICAL: Use serial processing for last few batches to avoid deadlocks
            # When only a handful of batches remain, executor-based parallelism can hang
            if num_batches < 5:
                print(f"🔒 Switching to SERIAL processing for last {num_batches} batches (prevents deadlock)")
                
                # Process serially with strict timeout
                for i, batch in enumerate(chunk_batches):
                    batch_idx = chunk_start + i
                    print(f"Processing batch {batch_idx+1}/{estimated_total_batches} (serial mode)...")
                    
                    try:
                        import signal
                        
                        # Use signal-based timeout for serial processing
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Batch {batch_idx+1} timed out after 300s")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # 5-minute timeout per batch
                        
                        try:
                            narrative, events = gpt5.analyze_frames(
                                batch, 
                                selected_profile, 
                                gpt5.context_state
                            )
                        finally:
                            signal.alarm(0)  # Cancel alarm
                        
                        if narrative.startswith("Error during analysis"):
                            error_logger.log_error("batch_processing", job_id, video_filename,
                                                 f"Serial batch {batch_idx+1} failed: {narrative}", None)
                            db.mark_error(job_id, f"Batch {batch_idx+1} failed: {narrative}")
                            print(f"❌ Batch {batch_idx+1} failed: {narrative}")
                            raise Exception(narrative)
                        
                        batch_results.append((batch_idx, narrative, events))
                        completed_batches += 1
                        
                        # Update progress
                        if estimated_total_batches:
                            db.update_progress(job_id, completed_batches, estimated_total_batches)
                            db.update_stage(job_id, 'analyzing_frames',
                                          progress_details=f"Analyzing batch {completed_batches}/{estimated_total_batches} (serial)")
                            print(f"✅ Batch {batch_idx+1} complete ({completed_batches}/{estimated_total_batches})")
                    
                    except TimeoutError as te:
                        error_logger.log_error("batch_processing", job_id, video_filename,
                                             f"Serial batch timeout: {str(te)}", te)
                        db.mark_error(job_id, f"Batch {batch_idx+1} timeout in serial mode")
                        print(f"❌ {str(te)}")
                        raise Exception(f"Serial batch timeout: {str(te)}")
                
                # Skip to context update (no parallel processing needed)
            else:
                # Parallel processing for larger chunks - Memory optimized for Replit
                max_workers = min(15, max(5, num_batches))
                print(f"Using {max_workers} workers for {num_batches} batches in this chunk (memory optimized)")
                
                # Timeout for each chunk (10 minutes max per chunk)
                chunk_timeout = 600
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(process_batch, (chunk_start + i, batch)): i 
                        for i, batch in enumerate(chunk_batches)
                    }
                    
                    try:
                        completed_futures = 0
                        
                        # CRITICAL: This catches the case where as_completed() hangs waiting for stuck futures
                        for future in as_completed(futures, timeout=chunk_timeout):
                            try:
                                # 3-minute timeout per batch (more aggressive than before)
                                batch_idx, narrative, events = future.result(timeout=180)
                                completed_futures += 1
                            except TimeoutError as batch_timeout:
                                error_logger.log_error("batch_processing", job_id, video_filename,
                                                     f"Individual batch timeout after 180s", batch_timeout)
                                db.mark_error(job_id, "Batch processing timeout - API call hung")
                                print(f"❌ Batch timed out after 180 seconds")
                                raise Exception("Batch API call timeout - OpenAI API not responding")
                            
                            if narrative.startswith("Error during analysis"):
                                error_logger.log_error("batch_processing", job_id, video_filename,
                                                     f"Analysis failed at batch {batch_idx+1}: {narrative}", None)
                                db.mark_error(job_id, f"Batch {batch_idx+1} analysis failed: {narrative}")
                                print(f"❌ Analysis failed at batch {batch_idx+1}: {narrative}")
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
                                db.update_progress(job_id, completed_batches, estimated_total_batches)
                                db.update_stage(job_id, 'analyzing_frames',
                                              progress_details=f"Analyzing batch {completed_batches}/{estimated_total_batches}")
                                if completed_batches % 5 == 0:
                                    error_logger.log_progress("frame_analysis", job_id, video_filename,
                                                            completed_batches, estimated_total_batches,
                                                            f"Frames processed: {total_frames_processed}")
                                    print(f"Progress: {completed_batches}/{estimated_total_batches} batches")
                            else:
                                db.update_progress(job_id, completed_batches, completed_batches)
                                db.update_stage(job_id, 'analyzing_frames',
                                              progress_details=f"Analyzing batch {completed_batches}")
                    except TimeoutError as timeout_error:
                        # This catches when as_completed() times out waiting for hung futures
                        hung_futures = [f for f in futures.keys() if not f.done()]
                        error_logger.log_error("batch_processing", job_id, video_filename,
                                             f"Chunk timeout - {len(hung_futures)} futures hung after {chunk_timeout}s", timeout_error)
                        db.mark_error(job_id, f"Chunk timeout - {len(hung_futures)} batches hung and did not complete")
                        print(f"❌ Chunk timed out with {len(hung_futures)} hung futures after {chunk_timeout}s")
                        
                        # Cancel all hung futures
                        for f in hung_futures:
                            f.cancel()
                        
                        raise Exception(f"Chunk timeout - {len(hung_futures)} API calls hung and did not respond")
            
            # Sort results by batch index and update context
            batch_results.sort(key=lambda x: x[0])
            for _, narrative, events in batch_results:
                gpt5.update_context(narrative, events)
            
            # CHECKPOINT: Save incremental progress after each chunk
            # This prevents token waste if job fails later
            partial_transcript = gpt5.get_full_transcript()
            if partial_transcript and completed_batches % 30 == 0:  # Save every 30 batches
                db.save_frame_transcript(job_id, partial_transcript)
                print(f"💾 Checkpoint saved: {completed_batches} batches completed")
                error_logger.log_info("checkpoint", job_id, video_filename,
                                    f"Saved checkpoint at batch {completed_batches}/{estimated_total_batches}")
            
            # Cleanup chunk batches and trigger garbage collection
            del chunk_batches
            del batch_results
            gc.collect()
            
            # Log memory usage after chunk and GC
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            print(f"📊 Memory after chunk {chunk_index}: {mem_after:.1f} MB RSS (Δ {mem_delta:+.1f} MB)")
            
            # Warn if memory is getting high (approaching 1GB)
            if mem_after > 900:
                print(f"⚠️  WARNING: Memory usage high ({mem_after:.1f} MB). Replit may OOM kill soon.")
        
        # If no frames were extracted, warn but don't fail
        if not frames_extracted_flag:
            print("⚠️ No frames were extracted from video")
            db.update_stage(job_id, 'frames_extracted')
        
        print("✅ Frame analysis complete!")
        error_logger.log_stage_exit("frame_analysis", job_id, video_filename, success=True)
        
        # Save final transcript (might already be saved from last checkpoint)
        transcript = gpt5.get_full_transcript()
        db.save_frame_transcript(job_id, transcript)
        print(f"💾 Final frame transcript saved: {len(transcript)} characters")
        db.update_stage(job_id, 'frames_analyzed', progress_details="All frames analyzed successfully")
        
        # Phase 5: Create enhanced narrative WITH TIMEOUT PROTECTION
        print("✨ Phase 5: Narrative Synthesis (AI)")
        events = gpt5.get_event_timeline()
        
        def create_narrative():
            """Wrapper for narrative creation with proper error checking."""
            db.update_stage(job_id, 'creating_narrative', 
                          progress_details="Creating enhanced narrative from video analysis")
            narrative = gpt5.create_enhanced_narrative(
                transcript=transcript,
                events=events,
                audio_transcript=audio_transcript,
                profile=selected_profile
            )
            if narrative.startswith("Error during"):
                raise Exception(f"Narrative enhancement failed: {narrative}")
            return narrative
        
        # Execute with 10-minute timeout and heartbeat monitoring
        enhanced_narrative = run_stage_with_timeout(
            stage_name='creating_narrative',
            timeout_seconds=600,  # 10 minutes max
            callable_func=create_narrative,
            db=db,
            job_id=job_id,
            error_logger=error_logger,
            video_filename=video_filename
        )
        
        db.save_narrative(job_id, enhanced_narrative)
        db.update_stage(job_id, 'narrative_created', progress_details="Enhanced narrative created successfully")
        narrative_file = save_analysis_file(
            enhanced_narrative, 'narrative', video_filename, job_id
        )
        print(f"📄 Narrative saved: {narrative_file}")
        
        # Phase 6: Generate medical assessment (if applicable) WITH TIMEOUT PROTECTION
        assessment_report = ""
        if profile == "Medical Assessment":
            print("📊 Phase 6: Medical Assessment (AI)")
            
            def create_assessment():
                """Wrapper for assessment creation with proper error checking."""
                db.update_stage(job_id, 'generating_assessment', 
                              progress_details="Generating medical assessment report")
                assessment = gpt5.assess_standardized_patient_encounter(enhanced_narrative)
                if assessment.startswith("Error during"):
                    raise Exception(f"Assessment generation failed: {assessment}")
                return assessment
            
            # Execute with 5-minute timeout and heartbeat monitoring
            assessment_report = run_stage_with_timeout(
                stage_name='generating_assessment',
                timeout_seconds=300,  # 5 minutes max
                callable_func=create_assessment,
                db=db,
                job_id=job_id,
                error_logger=error_logger,
                video_filename=video_filename
            )
            
            db.save_assessment(job_id, assessment_report)
            assessment_file = save_analysis_file(
                assessment_report, 'assessment', video_filename, job_id
            )
            print(f"📄 Assessment saved: {assessment_file}")
            
            # Phase 7: Generate PDF report WITH TIMEOUT PROTECTION
            print("📋 Phase 7: PDF Generation & Upload to Object Storage")
            from pdf_generator import create_assessment_pdf
            from storage_manager import StorageManager
            
            output_dir = get_output_dir(video_filename, job_id)
            pdf_path = output_dir / "report.pdf"
            
            def create_pdf():
                """Wrapper for PDF generation and upload to object storage."""
                db.update_stage(job_id, 'generating_pdf', 
                              progress_details="Generating PDF report")
                create_assessment_pdf(assessment_report, str(pdf_path))
                db.save_pdf_path(job_id, str(pdf_path))
                db.save_output_dir(job_id, str(output_dir))
                
                # Upload PDF to object storage for persistence
                db.update_stage(job_id, 'uploading_pdf',
                              progress_details="Uploading PDF to cloud storage")
                storage = StorageManager()
                storage_key = storage.upload_pdf(str(pdf_path), job_id, video_filename)
                
                if storage_key:
                    db.save_storage_key(job_id, storage_key)
                    print(f"☁️  PDF uploaded to object storage: {storage_key}")
                else:
                    print(f"⚠️  PDF upload failed, local copy saved: {pdf_path}")
                
                return str(pdf_path)
            
            try:
                # Execute with 3-minute timeout and heartbeat monitoring
                pdf_result = run_stage_with_timeout(
                    stage_name='generating_pdf',
                    timeout_seconds=180,  # 3 minutes max
                    callable_func=create_pdf,
                    db=db,
                    job_id=job_id,
                    error_logger=error_logger,
                    video_filename=video_filename
                )
                
                print(f"📄 PDF report saved: {pdf_result}")
                error_logger.log_info("pdf_generation", job_id, video_filename,
                                     f"PDF report generated successfully: {pdf_result}")
            except Exception as pdf_error:
                error_logger.log_error("pdf_generation", job_id, video_filename,
                                      f"PDF generation failed: {str(pdf_error)}", pdf_error)
                print(f"⚠️  PDF generation failed (non-critical): {str(pdf_error)}")
        else:
            db.update_stage(job_id, 'narrative_complete', status='completed',
                          progress_details="Analysis completed successfully")
        
        # Cleanup temp files
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
            error_logger.log_info("cleanup", job_id, video_filename,
                                 f"Cleaned up audio file: {audio_path}")
        if Path(video_path).exists():
            Path(video_path).unlink()
            error_logger.log_info("cleanup", job_id, video_filename,
                                 f"Cleaned up video file: {video_path}")
        
        error_logger.log_stage_exit("processing", job_id, video_filename, success=True)
        error_logger.log_info("job_completion", job_id, video_filename,
                             f"Analysis completed successfully - Total batches: {completed_batches}, Total frames: {total_frames_processed}")
        print("🎉 Analysis completed successfully!")
        print("🗑️ Temporary files cleaned up")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Analysis failed: {error_msg}")
        print(traceback.format_exc())
        error_logger.log_error("processing", job_id, video_filename,
                             f"Analysis failed with error: {error_msg}", e)
        error_logger.log_stage_exit("processing", job_id, video_filename, success=False)
        
        db.mark_error(job_id, error_msg)
        
        if video_path and Path(video_path).exists():
            Path(video_path).unlink()
            error_logger.log_info("cleanup", job_id, video_filename,
                                 f"Cleaned up video file after error: {video_path}")
        
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
            error_logger.log_info("cleanup", job_id, video_filename,
                                 f"Cleaned up audio file after error: {audio_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_video_background.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    process_video_job(job_id)
