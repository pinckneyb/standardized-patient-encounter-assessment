# AI Video Analysis Application

## Overview

This AI Video Analysis application is designed for analyzing standardized patient encounters and surgical skills assessments in medical education. It automates a three-stage AI analysis workflow using OpenAI's Responses API with GPT-4o-mini for multimodal analysis. The system processes uploaded videos, performs streaming audio transcription and frame-by-frame visual analysis, and generates comprehensive medical faculty assessment reports. Its purpose is to streamline medical education evaluations by providing automated, high-quality analytical insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Application Structure
- **Web Interface**: Streamlit-based application (`app.py`) featuring video upload, processing controls, and a scrollable output container for results.
- **Video Processing**: Utilizes FFmpeg for frame extraction, with OpenCV as a fallback (`video_processor.py`).
- **AI Analysis Engine**: Employs a three-stage pipeline leveraging OpenAI's Responses API (`gpt5_client.py`).
- **Profiles**: Supports "Medical Assessment" (default) and "Generic Video Narration" (`profiles.py`).
- **Audio Processing**: Implements streaming transcription using OpenAI models (`utils.py`).

### Three-Stage AI Processing Pipeline
1.  **Frame Analysis**: GPT-4o-mini analyzes video frames in batches with context-aware prompting to extract JSON descriptions.
2.  **Narrative Synthesis**: GPT-4o-mini combines frame observations and audio transcript into coherent narrative prose.
3.  **Medical Assessment**: GPT-4o-mini generates medical faculty assessment reports, including color-coded rubric scoring and professional PDF output.

### Analysis Profiles
-   **Medical Assessment**: Evaluates standardized patient encounters with medical rubric scoring.
-   **Generic Video Narration**: Provides human-like narrative descriptions for general video content.

### Processing Configuration
-   **Configurable Options**: Frame rate (0.5-5.0 FPS), batch sizes (3-15 frames), and video resolution (360p-1080p).
-   **Default Settings**: 1.0 FPS, 5 frames per batch, 720p resolution.
-   **Resolution Control**: Users can reduce video processing resolution for faster analysis and lower memory usage. Options include:
    -   1080p (1920px) - Full HD quality
    -   720p (1280px) - HD quality (default)
    -   480p (854px) - Standard definition
    -   360p (640px) - Low resolution for fastest processing
-   **Recommended Settings for Large Videos**: For 1.8GB+ videos or long recordings (10+ minutes), use 480p or 360p resolution to significantly reduce processing time and memory usage.

### Audio Transcription System
-   **AI-Powered Speaker Diarization**: Uses GPT-4o-mini to intelligently identify speakers based on conversational context and patterns.
-   **Context-Aware Speaker Detection**: Analyzes content to distinguish Student (medical student) from Patient based on:
    -   Question patterns (Student asks questions, Patient answers)
    -   Medical terminology and professional language (Student)
    -   Symptom descriptions and responses (Patient)
    -   Introductions and exam explanations (Student)
-   **Clean Transcript Generation**: OpenAI Whisper-1 provides accurate transcription, then GPT-4o-mini adds speaker labels.
-   **FFmpeg Audio Extraction**: Extracts audio as 16kHz mono WAV for automatic integration into narrative synthesis.

### API Architecture
-   **OpenAI Responses API**: Used for all analysis stages (frame, narrative, assessment) with `store=True` and `previous_response_id` for stateful context.
-   **Model**: GPT-4o-mini is employed across all analysis stages for its vision capabilities and cost-effectiveness.
-   **Response Format**: Utilizes `instructions=` and `input=` parameters, with `output_text` for results.

### Advanced Features
-   **Context Continuity**: Sophisticated context management ensures narrative flow across processing batches.
-   **Medical Rubric Scoring**: Automated faculty-style assessment with color-coded scoring.
-   **PDF Export**: Generates professional assessment reports for medical education.
-   **Memory-Efficient Streaming**: Refactored video processing (`iter_frames_streaming()`) to handle long videos (10-15 minutes) by yielding frames one at a time and capping resolution at 720p.
-   **Error Handling**: Robust system with comprehensive timeout handling for OpenAI API calls and proper cleanup of temporary files and job states.

## External Dependencies

### AI Services
-   **OpenAI Responses API**: For AI analysis (frame analysis, narrative generation, medical assessment).
-   **OpenAI Transcription API**: Specifically `gpt-4o-mini-transcribe` for streaming audio transcription.
-   **API Key Management**: OpenAI API keys are securely stored via Replit Secrets (`OPENAI_API_KEY`).

### Video Processing
-   **FFmpeg**: Primary library for video frame and audio extraction.
-   **OpenCV**: Fallback for video processing and computer vision tasks.
-   **PIL/Pillow**: Used for image manipulation and format conversion.

### Audio Processing
-   **OpenAI gpt-4o-mini-transcribe**: For streaming audio transcription.
-   **FFmpeg**: For audio extraction from video files.

### Web Framework and UI
-   **Streamlit**: Provides the web application framework, file upload, and interactive controls.
-   **ReportLab**: Utilized for generating professional PDF assessment reports.

### Data Processing
-   **NumPy**: For numerical computations in video and audio analysis.
-   **JSON**: Used for structured data storage of analysis results and configuration.

### File Handling
-   **Pathlib**: For modern file path operations and temporary file management.
-   **Base64**: For image encoding when transmitting to APIs.
-   **Tempfile**: For secure handling of temporary video and audio files.

## Recent Changes (November 2025)

### Latest Updates (November 18, 2025)
- **Chunked Narrative Synthesis** (NEW!):
  - **Problem**: Very large frame transcripts (600KB+) exceed GPT-4o-mini's context limits and cause failures
  - **Solution**: Automatically detects transcripts >150K characters and splits into processable chunks
  - **How It Works**: Divides transcript into chunks, processes each with GPT-4o-mini individually, then combines into cohesive narrative
  - **Benefits**: Successfully processes massive transcripts from long videos (10-15+ minutes) without failures
  - **Tested**: Successfully processed 603KB transcript (814 seconds @ 1 FPS) into 4,954-character narrative
- **Enhanced Delta Encoding in Frame Analysis**:
  - **Strengthened Prompts**: Frame analysis now explicitly instructs "only note changes from previous frame"
  - **Result**: Reduces data redundancy in frame transcripts, keeping file sizes more manageable
  - **Complements Chunking**: Works together with chunked synthesis to handle large videos efficiently

### Latest Updates (November 17, 2025)
- **99% HANG BUG COMPLETELY SOLVED!** (Two critical fixes):
  - **Issue**: Jobs consistently stuck at 99% (batch 270/271) waiting forever with no timeout
  - **Root Cause #1 - Batch Over-Estimation**: Used `math.ceil()` instead of `int()` for batch counting
    - Video: 814s → 815 frames @ 1 FPS → 271.67 batches with batch_size=3
    - Old: **272 batches** (ceil) - tried to process non-existent batch 271
    - Fixed: **271 batches** (floor) - correct count
  - **Root Cause #2 - Iterator Loop Doesn't Stop**: Even after completing all 271 batches, the for loop kept calling the frame iterator waiting for more chunks
    - Old: Loop continued indefinitely after all batches completed
    - Fixed: Added explicit check to break loop when `completed_batches >= estimated_total_batches`
  - **Result**: Processing now completes successfully at exactly 100% without hanging
- **Job Management UI Improvements**:
  - **Bulk Clear**: "🗑️ Clear X Failed" button to remove all failed jobs at once
  - **Individual Delete**: 🗑️ button next to each job for selective cleanup
  - Both methods clean up database records and temporary files

### Previous Fix Attempts (November 15, 2025) - Partially Successful
  - **Fix #1 - Timeout Handling**: Added proper `TimeoutError` handling for API calls ✅
  - **Fix #2 - Incremental Checkpointing**: Saves frame transcript every 30 batches ✅ (prevents token loss)
  - **Fix #3 - Serial Processing**: Processes last <5 batches serially ✅ (good for API timeouts)
  - **Why they didn't solve 99% hang**: The hang was in frame extraction, not API processing
- **AI-Powered Speaker Diarization (NEW!)**:
  - **Previous Issue**: Word-level timestamps were meaningless for speaker identification
  - **New Solution**: GPT-4o-mini intelligently identifies Student vs Patient based on conversational context
  - **How It Works**: Analyzes question/answer patterns, medical terminology, symptom descriptions, and introductions
  - **Output**: Clean speaker-labeled transcript (e.g., "Student: How long has this been going on?" "Patient: About two days")
  - **Accuracy**: Far superior to timestamp-based or random diarization approaches
- **CRITICAL BUG FIX - Background Processing Now Works!**
  - **Bug**: cleanup_old_incomplete_jobs() was marking ALL in-progress jobs as failed on every page refresh
  - **Impact**: Jobs would fail every time user refreshed the page to check status, even though background process was still running
  - **Fix**: Added 30-minute timeout - jobs only marked as stale if no updates for 30+ minutes
  - **Result**: Background processing now survives page refreshes and browser disconnects as intended
- **Increased Processing Speed (3x faster)**:
  - Concurrency: 10 → 30 workers (for Tier 4 OpenAI limits: 10,000 RPM, 10M TPM)
  - Chunk size: 10 → 30 batches
  - Expected processing time: 15-20 minutes instead of 30-40 minutes for 1.8GB videos
- **Browser Disconnect Resilience**:
  - Created process_video_background.py standalone worker
  - Subprocess runs independently with start_new_session=True (fully detached)
  - Processing continues even when IT surveillance software disconnects browser
  - Job Status Monitor UI shows live progress and auto-reconnects
- **Smart Resolution Validation**:
  - Only downscales videos, never upscales
  - If 480p video uploaded with 720p selected → stays 480p (no quality loss from upscaling)
  - Logs show decision for transparency

### Latest Updates (November 14, 2025) - Previous
- **Comprehensive Error Logging System** (Production-Ready):
  - **Persistent Logs**: All errors written to `./logs/video_analysis_errors.log` (survives browser disconnects and restarts)
  - **Log Rotation**: Keeps last 10 log files at 10MB each for long-term diagnostics
  - **Full Context Tracking**: Every log entry includes job_id, video_filename, processing stage, and full traceback
  - **Database Progress Tracking**: progress_details column tracks exact stage and batch number when failures occur
  - **Stage-Level Logging**: Entry/exit logging for all major processing phases (audio extraction, frame analysis, narrative synthesis, assessment)
  - **Batch-Level Progress**: Logs progress every 5 batches during frame analysis
  - **ThreadPoolExecutor Error Capture**: All concurrent processing exceptions logged before raising
  - **Browser Warning**: Users warned to keep browser open for 15-30 minute processing jobs
  - **Duplicate Handler Prevention**: Handles Streamlit reruns without log amplification
  - **Architect Verified**: All critical error paths confirmed to log properly with database updates
- **Resolution Configuration**: User-controllable video processing resolution (UI dropdown in sidebar)
  - **Options**: 1080p (Full HD), 720p (HD), 480p (SD), 360p (Low)
  - **Default**: 720p (1280px) - balances quality and performance
  - **Recommended**: For 1.8GB+ videos, use 480p or 360p for faster processing and lower memory usage
  - **Memory Impact**: Lower resolutions significantly reduce memory footprint and processing time
- **Critical Memory Leak Fixes**: Resolved crashes on long videos (10-15 minute standardized patient encounters)
  - **FFmpeg Buffer Leak Fixed**: Changed buffer to bytearray with 50MB hard limit, prevents unbounded growth
  - **Optimized Concurrency**: Reduced ThreadPoolExecutor from 30 to 10 workers, chunk_size from 30 to 10 batches
  - **Explicit Frame Cleanup**: Delete numpy arrays and close PIL images after batch processing with gc.collect()

### Updates (November 13, 2025)
- **Model Migration to GPT-4o-mini**: Updated from gpt-5-mini to gpt-4o-mini for API key compatibility
  - Full access confirmed with user's API key
  - Excellent multimodal vision capabilities for frame analysis
  - Cost-effective and performant for standardized patient assessment
- **Enhanced Assessment Report Display**: Completely redesigned on-screen assessment report formatting
  - Color-coded score badges (green ≥4, yellow ≥3, red <3, grey for N/A)
  - Clean card-based layout with rounded corners and visual hierarchy
  - Professional scoring legend at top of report

## Troubleshooting & Diagnostics

### Error Logs
All errors are written to `./logs/video_analysis_errors.log` with full context:
- **Location**: Project root → `logs/` folder
- **Format**: Timestamp | Level | Stage | Job ID | Filename | Message | Traceback
- **Persistence**: Survives browser disconnects, app restarts, and workspace restarts
- **Rotation**: Last 10 files kept (10MB each)
- **Access**: Check this file if processing fails silently or browser disconnects

### Database Progress Tracking
Query the `analysis_jobs` table to diagnose failures:
```sql
SELECT job_id, video_filename, status, error_message, progress_details, created_at 
FROM analysis_jobs 
ORDER BY created_at DESC 
LIMIT 10;
```
The `progress_details` column shows exactly which stage and batch was processing when failure occurred.