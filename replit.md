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
-   **Word-Level Timestamps**: Uses OpenAI Whisper-1 for precise, timestamped transcription (e.g., "[HH:MM:SS] text").
-   **Medical Assessment Ready**: Facilitates identification of speakers (Student vs. Patient) via timestamped dialogue, avoiding speaker diarization inaccuracies.
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