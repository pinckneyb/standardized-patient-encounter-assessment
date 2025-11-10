# AI Video Analysis Application

## Overview

This is an AI Video Analysis application specialized for analyzing standardized patient encounters in medical education. The system implements a complete three-stage automated analysis workflow using OpenAI's latest Responses API with GPT-5 for improved performance and cost efficiency.

The system processes uploaded videos through an AI pipeline with streaming audio transcription and frame-by-frame visual analysis to generate comprehensive medical faculty assessment reports.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Application Structure
- **Main Application**: Streamlit-based web interface (`app.py`) with scrollable output container, video upload, processing controls, and analysis results
- **Video Processing Pipeline**: FFmpeg-based frame extraction with OpenCV fallback (`video_processor.py`)
- **AI Analysis Engine**: Three-stage processing using OpenAI's Responses API (`gpt5_client.py`)
- **Profile System**: Medical Assessment (default) and Generic Video Narration profiles (`profiles.py`)
- **Audio Processing**: Streaming transcription with latest OpenAI models (`utils.py`)

### Three-Stage AI Processing Pipeline
The system employs a sophisticated three-pass analysis approach:
1. **Stage 1 (Frame Analysis)**: GPT-5 analyzes video frames in batches (default 5 frames) with context-aware prompting to extract JSON descriptions
2. **Stage 2 (Narrative Synthesis)**: GPT-5 combines frame observations and audio transcript into coherent narrative prose with enhanced storytelling
3. **Stage 3 (Medical Assessment)**: GPT-5 generates medical faculty assessment report with color-coded rubric scoring and professional PDF output

### Analysis Profiles
- **Medical Assessment** (default): Standardized patient encounter evaluation with medical rubric scoring
- **Generic Video Narration**: Human-like narrative description for general video content

### Processing Configuration
- **Default Settings**: 1.0 FPS, batch size of 5 frames
- **Configurable Options**: FPS (0.5-5.0), batch sizes (3-15 frames)
- **Scrollable Output**: All analysis output displays in scrollable container below upload widget

### Audio Transcription System
- **Word-Level Timestamps**: OpenAI Whisper-1 transcription with precise timestamps for each phrase
- **Timestamp Format**: Transcripts formatted as "[HH:MM:SS] text" for easy speaker identification by content
- **Medical Assessment Ready**: Faculty can identify Student vs Patient by reviewing timestamped dialogue
- **No AI Attribution Errors**: Avoids speaker diarization inaccuracies for reliable medical grading
- **FFmpeg Audio Extraction**: Extracts audio from video as 16kHz mono WAV
- **Automatic Integration**: Audio transcript automatically included in narrative synthesis

### API Architecture
- **Responses API**: Migrated from Chat Completions to new Responses API
- **Stateful Context**: Uses `store=True` and `previous_response_id` for response chaining
- **Model**: GPT-5 (released August 7, 2025) for all analysis stages
- **Performance**: 3% better accuracy, 40-80% cost reduction vs Chat Completions
- **Response Format**: Uses `instructions=` and `input=` parameters, `output_text` for results

### Advanced Features
- **Context Continuity**: Maintains narrative flow across batches using sophisticated context management
- **Enhanced Narratives**: GPT-5 processed coherent storytelling with absolute specificity
- **Medical Rubric Scoring**: Automated faculty-style assessment with color-coded scoring
- **PDF Export**: Professional assessment reports for medical education

## External Dependencies

### AI Services
- **OpenAI Responses API**: GPT-5 for all analysis stages (frame analysis, narrative, assessment)
- **OpenAI Transcription API**: gpt-4o-mini-transcribe for streaming audio transcription
- **API Key Management**: Securely stored via Replit Secrets (OPENAI_API_KEY)

### Video Processing
- **FFmpeg**: Primary video processing library for frame and audio extraction
- **OpenCV**: Fallback video processing and computer vision operations
- **PIL/Pillow**: Image manipulation and format conversion

### Audio Processing
- **OpenAI gpt-4o-mini-transcribe**: Streaming audio transcription with confidence scores
- **FFmpeg**: Audio extraction from video files (16kHz mono WAV)
- **PyAnnote Audio** (optional): Speaker diarization and identification
- **TensorFlow/TensorFlow Hub** (optional): Audio analysis with YAMNet

### Web Framework and UI
- **Streamlit**: Web application framework with file upload, progress tracking, and interactive controls
- **Scrollable Container**: Custom CSS for 600px max-height output area
- **ReportLab**: PDF report generation for medical assessments

### Data Processing
- **NumPy**: Numerical computations for video and audio analysis
- **JSON**: Structured data storage for analysis results and configuration

### File Handling
- **Pathlib**: Modern file path operations and temporary file management
- **Base64**: Image encoding for API transmission
- **Tempfile**: Secure temporary video and audio file handling

## Recent Changes (October 2025)

### Latest Updates (October 8, 2025)
- **Timestamp-Based Transcription**: Disabled automatic speaker diarization in favor of accurate timestamps
  - Word-level timestamps using OpenAI Whisper-1: "[HH:MM:SS] text" format
  - Faculty can identify Student vs Patient by reviewing timestamped dialogue content
  - Avoids AI attribution errors from inaccurate speaker diarization
  - More reliable for medical education assessment and grading
  - Faster processing (~30 seconds saved per video)
- **Tier 4 Optimization**: Increased concurrency for faster processing
  - Concurrent batches increased from 10 to 30 (3x improvement)
  - Default batch size reduced from 5 to 3 frames (more parallelism)
  - Optimized for Tier 4 OpenAI limits (10,000 RPM, 10M TPM)
  - Uses only ~18% of rate limits with plenty of headroom
- **Automatic File Export System**: Analysis results now automatically saved as text files
  - Audio transcripts saved to `transcripts/` folder
  - Enhanced narratives saved to `narratives/` folder
  - Assessment reports saved to `assessments/` folder
  - Files named with video name, job ID, and timestamp for easy organization
  - All output folders excluded from git for clean repository

### Updates (October 6, 2025)
- **Resume System with Database Persistence**: Implemented complete analysis job tracking
  - PostgreSQL database stores job state, progress, and all analysis results
  - Jobs automatically marked as in_progress/completed with proper state transitions
  - State persisted at each processing stage (audio, frames, narrative, assessment)
  - Batch processing progress tracked in real-time
  - Failed jobs marked with error messages for debugging
  - UI displays incomplete jobs with resume capability (foundation for future full resume)
- **Automatic Cleanup System**: Intelligent temporary file management
  - Temp video and audio files cleaned up after successful completion
  - Cleanup also performed on analysis errors
  - Files removed from filesystem and marked in database
  - Prevents disk space accumulation from abandoned analyses
- **Resilient Architecture**: System prepared for browser disconnection recovery
  - All analysis data persisted to database
  - Job state tracked throughout pipeline
  - Foundation ready for full resume implementation
- **Concurrent Batch Processing**: Implemented parallel batch processing for maximum performance
  - Processes 10 batches concurrently using ThreadPoolExecutor
  - Batches within each chunk of 10 share the same starting context
  - Context is updated after all 10 batches complete, then moves to next chunk
  - Leverages Tier 4 API limits for faster processing
- **Professional PDF Export**: Implemented ReportLab-based PDF generation for assessment reports
  - Color-coded scoring tables (green ≥4, yellow ≥3, red <3, grey for N/A)
  - Professional medical report formatting with scoring legend
  - Word-wrapped feedback text using Paragraph objects for proper formatting
  - Robust JSON parsing handles markdown wrapping and plain text fallback
  - One-click download button in assessment report section
  - Type-safe handling for both dict and string category data
- **Robust Error Handling**: Enhanced PDF generation with comprehensive edge case handling
  - Safely handles missing scores (displays as "N/A")
  - Handles mixed data types in assessment categories
  - Type-safe score conversion with fallback
  - Dynamic table styling only for existing rows
  - Proper text wrapping in feedback cells

### Previous Updates
- **Audio Streaming Integration**: Implemented streaming audio transcription with gpt-4o-mini-transcribe
- **Responses API Migration**: Full migration from Chat Completions to Responses API
- **Model Upgrade**: Upgraded from gpt-5-mini to gpt-5 for better performance
- **Default Profile**: Changed default to Medical Assessment profile
- **Default Batch Size**: Changed from 7 to 5 frames per batch
- **Profile Cleanup**: Removed Sports Commentary profile
- **API Key Persistence**: Configured Replit Secrets for permanent API key storage
- **UI Improvements**: Scrollable output container prevents page scrolling

### API Integration
- **OpenAI Integration**: Configured via Replit blueprint for secure key management
- **Secret Storage**: OPENAI_API_KEY stored in Replit Secrets (AES-256 encrypted)
- **Automatic Loading**: API key automatically loads from environment on every run
