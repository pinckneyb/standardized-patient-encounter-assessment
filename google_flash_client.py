"""
Google Flash 2.5 client module for direct video analysis.
Replaces the OpenAI frame-by-frame approach with native video ingestion.
"""

import os
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from google import genai
from google.genai import types

class GoogleFlashClient:
    """Client for Google Gemini Flash 2.5 video analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Flash client.
        
        Args:
            api_key: Google API key (will use env var if not provided)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
        self.uploaded_file = None
    
    def upload_video(self, video_path: str, progress_callback: Optional[Callable[[str, int], None]] = None) -> str:
        """
        Upload video to Google's file API for processing.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback(status_message, progress_percent)
            
        Returns:
            File URI for the uploaded video
        """
        if progress_callback:
            progress_callback("Uploading video to Google...", 5)
        
        print(f"📤 Uploading video: {video_path}")
        
        self.uploaded_file = self.client.files.upload(file=video_path)
        
        while self.uploaded_file.state.name == "PROCESSING":
            if progress_callback:
                progress_callback("Video processing on Google servers...", 10)
            print("⏳ Waiting for video processing...")
            time.sleep(5)
            self.uploaded_file = self.client.files.get(name=self.uploaded_file.name)
        
        if self.uploaded_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {self.uploaded_file.state}")
        
        print(f"✅ Video uploaded successfully: {self.uploaded_file.name}")
        if progress_callback:
            progress_callback("Video ready for analysis", 15)
        
        return self.uploaded_file.name
    
    def analyze_video(
        self,
        profile_type: str,
        audio_transcript: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyze uploaded video using Flash 2.5.
        
        Args:
            profile_type: 'standardized_patient' or 'generic'
            audio_transcript: Optional audio transcript to include
            progress_callback: Optional callback(status_message, progress_percent)
            
        Returns:
            Dictionary with 'narrative' and 'raw_response' keys
        """
        if not self.uploaded_file:
            raise ValueError("No video uploaded. Call upload_video first.")
        
        if progress_callback:
            progress_callback("Analyzing video with AI...", 20)
        
        prompt = self._build_analysis_prompt(profile_type, audio_transcript)
        
        print(f"🤖 Starting Flash 2.5 video analysis...")
        print(f"   Profile: {profile_type}")
        print(f"   Transcript included: {bool(audio_transcript)}")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                self.uploaded_file,
                prompt
            ]
        )
        
        narrative = response.text
        
        if progress_callback:
            progress_callback("Video analysis complete", 60)
        
        print(f"✅ Analysis complete: {len(narrative)} characters")
        
        return {
            "narrative": narrative,
            "raw_response": response
        }
    
    def generate_assessment(
        self,
        narrative: str,
        audio_transcript: Optional[str] = None,
        rubric: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> str:
        """
        Generate medical faculty assessment based on narrative.
        
        Args:
            narrative: The video analysis narrative
            audio_transcript: Optional audio transcript
            rubric: Optional assessment rubric
            progress_callback: Optional callback(status_message, progress_percent)
            
        Returns:
            Assessment text
        """
        if progress_callback:
            progress_callback("Generating assessment...", 70)
        
        prompt = self._build_assessment_prompt(narrative, audio_transcript, rubric)
        
        print(f"📋 Generating medical assessment...")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt]
        )
        
        assessment = response.text
        
        if progress_callback:
            progress_callback("Assessment complete", 85)
        
        print(f"✅ Assessment complete: {len(assessment)} characters")
        
        return assessment
    
    def cleanup(self):
        """Delete uploaded video file from Google's servers."""
        if self.uploaded_file:
            try:
                self.client.files.delete(name=self.uploaded_file.name)
                print(f"🗑️ Deleted uploaded file: {self.uploaded_file.name}")
            except Exception as e:
                print(f"⚠️ Could not delete uploaded file: {e}")
            self.uploaded_file = None
    
    def _build_analysis_prompt(self, profile_type: str, audio_transcript: Optional[str] = None) -> str:
        """Build the analysis prompt based on profile type."""
        
        if profile_type == "standardized_patient":
            base_prompt = """You are an expert medical education evaluator observing a standardized patient encounter.

Analyze this video and provide a detailed clinical narrative that describes:

1. **Setting and Context**: Describe the clinical environment and initial setup
2. **Encounter Chronology**: Document the encounter in chronological order with timestamps
3. **Clinical Skills Observed**:
   - Communication skills (verbal and non-verbal)
   - History taking technique
   - Physical examination approach (if applicable)
   - Hand hygiene and infection control
   - Professional behavior and rapport building
4. **Patient Interaction Quality**:
   - How the student addresses the patient
   - Active listening behaviors
   - Empathy and patient-centered care
5. **Technical Observations**:
   - Any clinical equipment used
   - Documentation practices observed

Format your response as a flowing clinical narrative with embedded timestamps like [00:01:23].
Use objective, professional language suitable for medical faculty review.
Focus on observable behaviors rather than assumptions about intent."""

        else:  # generic profile
            base_prompt = """You are an expert video analyst. Analyze this video comprehensively.

Provide a detailed narrative that describes:

1. **Setting**: Describe the environment, location, and context
2. **Participants**: Describe each person visible (appearance, role, actions)
3. **Chronological Events**: Document what happens throughout the video with timestamps
4. **Key Observations**: Notable moments, interactions, or details
5. **Audio Content**: Any speech, sounds, or audio elements

Format your response as a flowing narrative with embedded timestamps like [00:01:23].
Be specific and objective in your descriptions."""

        if audio_transcript:
            base_prompt += f"""

---
AUDIO TRANSCRIPT (for reference - align visual observations with this):
{audio_transcript}
---"""

        return base_prompt
    
    def _build_assessment_prompt(
        self, 
        narrative: str, 
        audio_transcript: Optional[str] = None,
        rubric: Optional[str] = None
    ) -> str:
        """Build the assessment prompt for medical faculty evaluation."""
        
        prompt = """You are a medical faculty member evaluating a standardized patient encounter.

Based on the following observational narrative, generate a professional assessment report.

## Assessment Format:

### 1. Executive Summary
Brief overview of the encounter and overall performance (2-3 sentences)

### 2. Strengths Observed
List specific positive behaviors with evidence from the narrative

### 3. Areas for Improvement
List specific areas needing development with evidence

### 4. Clinical Skills Assessment
Rate each area (Excellent/Good/Satisfactory/Needs Improvement):
- **Communication**: [Rating] - Brief justification
- **History Taking**: [Rating] - Brief justification  
- **Physical Examination**: [Rating] - Brief justification (if applicable)
- **Professionalism**: [Rating] - Brief justification
- **Patient Safety**: [Rating] - Brief justification

### 5. Specific Recommendations
Actionable suggestions for improvement

### 6. Overall Assessment
Final summary with recommendation (Pass/Needs Review/Remediation Recommended)

---
OBSERVATIONAL NARRATIVE:
"""
        prompt += narrative
        
        if audio_transcript:
            prompt += f"""

---
AUDIO TRANSCRIPT:
{audio_transcript}"""

        if rubric:
            prompt += f"""

---
ASSESSMENT RUBRIC (use for scoring):
{rubric}"""

        return prompt


def test_google_flash_client():
    """Quick test of the Google Flash client."""
    try:
        client = GoogleFlashClient()
        print("✅ Google Flash client initialized successfully")
        print(f"   Model: {client.model}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False


if __name__ == "__main__":
    test_google_flash_client()
