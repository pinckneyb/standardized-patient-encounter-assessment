# AI Video Analysis Application

## Overview

This AI Video Analysis application analyzes standardized patient encounters and surgical skills assessments in medical education. It uses **Google Gemini Flash 2.5** for direct video ingestion and analysis (no frame extraction needed), combined with OpenAI Whisper for audio transcription. The system generates comprehensive medical faculty assessment reports with professional PDF output.

## User Preferences

Preferred communication style: Simple, everyday language.

## Architecture Change (December 2024)

**Major Migration: OpenAI Frame-Based → Google Flash 2.5 Direct Video**

Previous architecture (deprecated):
- Extract frames from video at configurable FPS
- Batch process frames with OpenAI GPT-4o-mini
- Assemble frame analyses into narrative

New architecture (current):
- Upload video directly to Google Flash 2.5 API
- Single API call for holistic video analysis
- Much simpler, faster, and more accurate

Key files changed:
- `google_flash_client.py` - New client for Flash 2.5 video analysis
- `process_video_flash.py` - Simplified background processor
- `app.py` - Updated UI (removed fps/batch_size/resolution controls)

Legacy files (kept for reference):
- `gpt5_client.py` - Original OpenAI client
- `process_video_background.py` - Original frame-based processor
- `video_processor.py` - Frame extraction logic

## Deployment Requirements

This application can run on **Autoscale** deployment (no longer requires Reserved VM).

### Why Autoscale Works Now
- Google Flash 2.5 handles video processing on their servers
- No local frame extraction or heavy compute needed
- Only audio transcription happens locally (quick)

### Storage Considerations
- **PostgreSQL Database**: Stores all text data (transcripts, narratives, assessments)
- **Replit Object Storage**: Stores PDF reports persistently in cloud storage
- Temporary files (uploaded videos, audio files) are cleaned up after processing

### Current Configuration
- **Deployment Type**: Autoscale
- **Run Command**: Streamlit on port 5000 with headless mode
- **Database**: PostgreSQL (persistent across deployments)
- **Storage**: Temporary files cleaned up after job completion

## System Architecture

### Core Application Structure
- **Web Interface**: Streamlit-based application (`app.py`) for video upload, profile selection, and results display.
- **Video Processing**: Google Flash 2.5 for direct video analysis (`google_flash_client.py`).
- **Audio Processing**: OpenAI Whisper for transcription (`audio_utils.py`).
- **Background Worker**: Simplified pipeline (`process_video_flash.py`).
- **PDF Generation**: ReportLab for professional reports (`pdf_generator.py`).

### Processing Pipeline (Flash 2.5)
1. **Audio Extraction**: FFmpeg extracts audio track from video
2. **Audio Transcription**: OpenAI Whisper-1 transcribes speech
3. **Video Upload**: Video uploaded to Google servers
4. **Video Analysis**: Flash 2.5 analyzes full video with audio context
5. **Assessment Generation**: Flash 2.5 generates medical faculty assessment
6. **PDF Generation**: ReportLab creates professional PDF report
7. **Cloud Upload**: PDF uploaded to Replit Object Storage

### Analysis Profiles
- **Medical Assessment**: Evaluates standardized patient encounters with rubric scoring
- **Generic Video Narration**: Provides general video analysis

## External Dependencies

### AI Services
- **Google Gemini Flash 2.5**: Direct video analysis (upload and analyze entire videos)
- **OpenAI Whisper-1**: Audio transcription

### Required API Keys
- `GOOGLE_API_KEY`: For Flash 2.5 video analysis
- `OPENAI_API_KEY`: For Whisper audio transcription

### Audio Processing
- **FFmpeg**: Audio extraction from video files

### Web Framework and UI
- **Streamlit**: Web application framework
- **ReportLab**: PDF report generation

### Storage and Persistence
- **PostgreSQL**: Job data, transcripts, narratives, assessments
- **Replit Object Storage**: PDF reports (persistent across deployments)

### Python Dependencies
- `google-genai`: Google Gemini API client
- `openai`: OpenAI API client (Whisper)
- `streamlit`: Web interface
- `reportlab`: PDF generation
- `psycopg2-binary`: PostgreSQL driver
- `replit-object-storage`: Cloud storage client
