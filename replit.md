# AI Video Analysis Application

## Overview

This AI Video Analysis application analyzes standardized patient encounters and surgical skills assessments in medical education. It automates a three-stage AI analysis workflow using OpenAI's Responses API with GPT-4o-mini for multimodal analysis. The system processes uploaded videos, performs streaming audio transcription and frame-by-frame visual analysis, and generates comprehensive medical faculty assessment reports, streamlining medical education evaluations.

## User Preferences

Preferred communication style: Simple, everyday language.

## Deployment Requirements

**CRITICAL: This application MUST be deployed as a Reserved VM, NOT run in development workspace.**

### Why Reserved VM is Required
- **Development workspaces** are designed for interactive coding and testing, NOT long-running compute tasks
- Video processing jobs (10-15 minutes for a 13-minute video) get killed by workspace resource limits
- Jobs will fail silently after ~3 minutes of processing in development mode

### Reserved VM Benefits
- ✅ **Dedicated computing resources** - No resource contention or kills
- ✅ **99.9% uptime guarantee** - Jobs complete reliably
- ✅ **No timeout limits** - Can process videos of any length
- ✅ **Designed for compute-intensive tasks** - Exactly what this app does

### Storage Considerations
- Reserved VMs have **no persistent file storage** - files are lost on republish
- ✅ **Already handled**: Application uses PostgreSQL database for all persistent data
- Output files (transcripts, narratives, PDFs) are stored in per-job directories (`outputs/`)
- Users should download results after processing completes

### Current Configuration
- **Deployment Type**: Reserved VM (configured in .replit)
- **Run Command**: Streamlit on port 5000 with headless mode
- **Database**: PostgreSQL (persistent across deployments)
- **Storage**: Temporary files cleaned up after job completion

## System Architecture

### Core Application Structure
- **Web Interface**: Streamlit-based application (`app.py`) for video upload, processing controls, and results display.
- **Video Processing**: Utilizes FFmpeg for frame extraction, with OpenCV as a fallback (`video_processor.py`).
- **AI Analysis Engine**: Employs a three-stage pipeline leveraging OpenAI's Responses API (`gpt5_client.py`).
- **Profiles**: Supports "Medical Assessment" (default) and "Generic Video Narration" (`profiles.py`).
- **Audio Processing**: Implements streaming transcription using OpenAI models (`utils.py`).

### Three-Stage AI Processing Pipeline
1.  **Frame Analysis**: GPT-4o-mini analyzes video frames in batches with context-aware prompting to extract JSON descriptions.
2.  **Narrative Synthesis**: GPT-4o-mini combines frame observations and audio transcript into clinical observational records using flat, undramatic language.
3.  **Medical Assessment**: GPT-4o-mini generates medical faculty assessment reports, including color-coded rubric scoring and professional PDF output.

### Analysis Profiles
-   **Medical Assessment**: Evaluates standardized patient encounters with medical rubric scoring.
-   **Generic Video Narration**: Provides human-like narrative descriptions for general video content.

### Processing Configuration
-   **Configurable Options**: Frame rate (0.5-5.0 FPS), batch sizes (3-15 frames), and video resolution (360p-1080p).
-   **Default Settings**: 1.0 FPS, 5 frames per batch, 720p resolution.
-   **Resolution Control**: Users can reduce video processing resolution for faster analysis and lower memory usage.
-   **Memory Optimization**: Frame analysis uses 15 concurrent workers (reduced from 50) to prevent OOM kills on Replit's memory-limited environment. This balances performance with reliability.

### Audio Transcription System
-   **AI-Powered Speaker Diarization**: Uses GPT-5 mini with Responses API to perform evidence-based speaker identification with stable speaker IDs, role tracking, and explicit evidence citations.
-   **Sophisticated Role Assignment**: Supports multiple roles (patient, student, resident, attending, nurse, tech, staff, family, unknown) based on explicit textual evidence only.
-   **Clean Transcript Generation**: OpenAI Whisper-1 provides accurate transcription (6-minute timeout for large files), then GPT-5 mini adds sophisticated speaker labels with confidence tracking (5-minute timeout).
-   **FFmpeg Audio Extraction**: Extracts audio as 16kHz mono WAV for narrative synthesis.

### API Architecture
-   **OpenAI Responses API**: Used for all analysis stages (frame, narrative, assessment, diarization) with `store=True` and `previous_response_id` for stateful context.
-   **Model**: GPT-4o-mini is employed for frame analysis, narrative, and assessment stages. GPT-5 mini is used for speaker diarization.

### Advanced Features
-   **Context Continuity**: Sophisticated context management ensures narrative flow.
-   **Medical Rubric Scoring**: Automated faculty-style assessment with color-coded scoring.
-   **PDF Export**: Generates professional assessment reports.
-   **Memory-Efficient Streaming**: Processes long videos by yielding frames one at a time and capping resolution at 720p.
-   **Error Handling**: Robust system with timeout handling for OpenAI API calls and proper cleanup.

## External Dependencies

### AI Services
-   **OpenAI Responses API**: For AI analysis (frame analysis, narrative generation, medical assessment, speaker diarization).
-   **OpenAI Transcription API**: Whisper-1 for streaming audio transcription.
-   **GPT-5 mini**: Fast, cost-efficient model for speaker diarization with evidence-based role assignment.
-   **GPT-4o-mini**: Used for frame analysis, narrative synthesis, and medical assessment generation.

### Video Processing
-   **FFmpeg**: Primary library for video frame and audio extraction.
-   **OpenCV**: Fallback for video processing and computer vision tasks.
-   **PIL/Pillow**: Used for image manipulation.

### Audio Processing
-   **OpenAI Whisper-1**: For streaming audio transcription.
-   **GPT-5 mini**: For sophisticated speaker diarization with evidence tracking.

### Web Framework and UI
-   **Streamlit**: Provides the web application framework.
-   **ReportLab**: Utilized for generating professional PDF assessment reports.

### Data Processing
-   **NumPy**: For numerical computations.
-   **JSON**: Used for structured data storage.

### File Handling
-   **Pathlib**: For file path operations.
-   **Base64**: For image encoding.
-   **Tempfile**: For secure handling of temporary files.