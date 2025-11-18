# AI Video Analysis Application

## Overview

This AI Video Analysis application analyzes standardized patient encounters and surgical skills assessments in medical education. It automates a three-stage AI analysis workflow using OpenAI's Responses API with GPT-4o-mini for multimodal analysis. The system processes uploaded videos, performs streaming audio transcription and frame-by-frame visual analysis, and generates comprehensive medical faculty assessment reports, streamlining medical education evaluations.

## User Preferences

Preferred communication style: Simple, everyday language.

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

### Audio Transcription System
-   **AI-Powered Speaker Diarization**: Uses GPT-4o-mini to intelligently identify speakers (Student vs Patient) based on conversational context.
-   **Clean Transcript Generation**: OpenAI Whisper-1 provides accurate transcription, then GPT-4o-mini adds speaker labels.
-   **FFmpeg Audio Extraction**: Extracts audio as 16kHz mono WAV for narrative synthesis.

### API Architecture
-   **OpenAI Responses API**: Used for all analysis stages (frame, narrative, assessment) with `store=True` and `previous_response_id` for stateful context.
-   **Model**: GPT-4o-mini is employed across all analysis stages.

### Advanced Features
-   **Context Continuity**: Sophisticated context management ensures narrative flow.
-   **Medical Rubric Scoring**: Automated faculty-style assessment with color-coded scoring.
-   **PDF Export**: Generates professional assessment reports.
-   **Memory-Efficient Streaming**: Processes long videos by yielding frames one at a time and capping resolution at 720p.
-   **Error Handling**: Robust system with timeout handling for OpenAI API calls and proper cleanup.

## External Dependencies

### AI Services
-   **OpenAI Responses API**: For AI analysis (frame analysis, narrative generation, medical assessment).
-   **OpenAI Transcription API**: Specifically `gpt-4o-mini-transcribe` for streaming audio transcription.

### Video Processing
-   **FFmpeg**: Primary library for video frame and audio extraction.
-   **OpenCV**: Fallback for video processing and computer vision tasks.
-   **PIL/Pillow**: Used for image manipulation.

### Audio Processing
-   **OpenAI gpt-4o-mini-transcribe**: For streaming audio transcription.

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