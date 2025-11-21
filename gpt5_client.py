"""
GPT-5 client module for video analysis using the new Responses API.
Migrated from Chat Completions API to Responses API for better performance and capabilities.
"""

import openai
from openai import APITimeoutError, APIConnectionError
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv
import httpx
from utils.error_logger import get_error_logger

load_dotenv()

class GPT5Client:
    """Client for GPT-5 Responses API integration with video analysis capabilities."""
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        """
        Initialize GPT-5 client with Responses API.
        
        Args:
            api_key: OpenAI API key (will use env var if not provided)
            timeout: Timeout in seconds for API calls (default 120 seconds)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=2  # Limit retries to prevent indefinite hanging
        )
        self.timeout = timeout
        
        self.context_state = ""
        self.full_transcript = []
        self.event_log = []
        self.previous_response_id = None
        
        self.error_logger = get_error_logger()
        self.job_id: Optional[str] = None
        self.video_filename: Optional[str] = None
    
    def set_job_context(self, job_id: str, video_filename: str):
        """Set job context for error logging."""
        self.job_id = job_id
        self.video_filename = video_filename
    
    def _call_api_with_retry(self, api_call_func, operation_name: str, max_retries: int = 3):
        """
        Call OpenAI API with exponential backoff retry logic for transient errors.
        
        Handles:
        - 500 server errors (retry with backoff)
        - 429 rate limit errors (retry with backoff)
        - Network timeouts (retry with backoff)
        
        Args:
            api_call_func: Lambda/function that makes the API call
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retry attempts (default 3)
            
        Returns:
            API response
            
        Raises:
            Exception: If all retries are exhausted
        """
        import time
        from openai import RateLimitError, InternalServerError
        
        for attempt in range(max_retries):
            try:
                return api_call_func()
            
            except (InternalServerError, RateLimitError) as e:
                # Transient errors that should be retried
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                    print(f"⚠️  {operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"   Retrying in {wait_time} seconds...")
                    self.error_logger.log_warning(
                        f"{operation_name}_retry", self.job_id, self.video_filename,
                        f"Retry attempt {attempt + 1}/{max_retries} after error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    self.error_logger.log_error(
                        operation_name, self.job_id, self.video_filename,
                        f"All {max_retries} retry attempts exhausted: {str(e)}", e
                    )
                    raise
            
            except Exception as e:
                # Non-retryable errors (invalid request, auth, etc.)
                self.error_logger.log_error(
                    operation_name, self.job_id, self.video_filename,
                    f"Non-retryable error: {str(e)}", e
                )
                raise
        
        raise Exception(f"{operation_name} failed after {max_retries} retries")
    
    def analyze_frames(self, frames: List[Dict], profile: Dict[str, Any], 
                      context_state: str = "") -> Tuple[str, List[Dict]]:
        """
        Analyze a batch of frames using GPT-5 Responses API.
        
        Args:
            frames: List of frame metadata dictionaries
            profile: Profile configuration for prompting
            context_state: Condensed context from previous analysis
            
        Returns:
            Tuple of (narrative_text, event_log)
        """
        try:
            # Type guard for profile
            if not isinstance(profile, dict):
                raise TypeError(f"profile must be a dict, got {type(profile).__name__}: {profile}")
            
            # Prepare the instruction (system prompt)
            instructions = profile['base_prompt']
            
            # Build the input content with context
            context_info = f"Context so far: {context_state if context_state else 'Beginning of video'}"
            timestamps = [f"{frame['timestamp_formatted']}" for frame in frames]
            timestamp_range = f"{timestamps[0]} to {timestamps[-1]}"
            
            input_text = f"""{context_info}

Analyze the following {len(frames)} frames from {timestamp_range}:
"""
            
            # Convert frames to base64 for API
            base64_frames = self._frames_to_base64(frames)
            
            # Create input with multimodal content (text + images)
            input_content = [
                {
                    "type": "input_text",
                    "text": input_text
                }
            ] + [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{frame_data}"
                } for frame_data in base64_frames
            ]
            
            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=[{
                    "role": "user",
                    "content": input_content
                }],
                store=True  # Enable stateful context for better performance
            )
            
            # Use the convenience property from Responses API
            if not response or not hasattr(response, 'output_text'):
                raise ValueError("Invalid API response: missing output_text")
            
            narrative_text = response.output_text
            
            if not narrative_text:
                raise ValueError("Invalid API response: empty output_text")
            
            # Store response ID for potential chaining
            self.previous_response_id = response.id
            
            # Ensure narrative text is properly encoded
            if isinstance(narrative_text, str):
                narrative_text = narrative_text.encode('utf-8', errors='replace').decode('utf-8')
            
            # Extract JSON events if present
            event_log = self._extract_events_from_response(narrative_text)
            
            return narrative_text, event_log
        
        except APITimeoutError as e:
            error_msg = f"OpenAI API request timed out after {self.timeout}s. The service may be overloaded or experiencing issues."
            self.error_logger.log_error("frame_analysis_api", self.job_id, self.video_filename,
                                       f"API Timeout: {error_msg}", e)
            return f"Error during analysis: {error_msg}", []
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API. Check your internet connection."
            self.error_logger.log_error("frame_analysis_api", self.job_id, self.video_filename,
                                       f"API Connection Error: {error_msg}", e)
            return f"Error during analysis: {error_msg}", []
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            self.error_logger.log_error("frame_analysis_api", self.job_id, self.video_filename,
                                       f"Unexpected error analyzing frames: {error_msg}", e)
            return f"Error during analysis: {error_msg}", []
    
    def _frames_to_base64(self, frames: List[Dict]) -> List[str]:
        """Convert frames to base64 strings."""
        import base64
        import io
        
        base64_frames = []
        for i, frame_data in enumerate(frames):
            try:
                # Convert PIL image to base64
                img_buffer = io.BytesIO()
                frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                base64_frames.append(img_str)
            except Exception as e:
                error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
                print(f"Error converting frame {i} to base64: {error_msg}")
                # Skip this frame and continue
                continue
        
        return base64_frames
    
    def _extract_events_from_response(self, response_text: str) -> List[Dict]:
        """Extract JSON events from GPT-5 response."""
        try:
            # Look for JSON blocks in the response
            json_pattern = r'```json\s*(\[.*?\])\s*```'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if matches:
                # Parse the first JSON match
                events = json.loads(matches[0])
                return events if isinstance(events, list) else []
            
            # Fallback: look for any JSON-like structure
            json_pattern2 = r'\[.*?\]'
            matches2 = re.findall(json_pattern2, response_text, re.DOTALL)
            
            for match in matches2:
                try:
                    events = json.loads(match)
                    if isinstance(events, list) and events:
                        return events
                except:
                    continue
            
            return []
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error extracting events: {error_msg}")
            return []
    
    def update_context(self, narrative: str, events: List[Dict]):
        """Update internal context with new analysis results."""
        self.full_transcript.append(narrative)
        self.event_log.extend(events)
        
        # Condense context for next batch
        if len(self.full_transcript) > 1:  # Only condense after first batch
            self.context_state = self.condense_context(self.full_transcript)
        else:
            self.context_state = "Beginning of video analysis"
    
    def reset_context(self):
        """Reset context state for new video."""
        self.context_state = ""
        self.full_transcript = []
        self.event_log = []
        self.previous_response_id = None
    
    def rescan_segment(self, start_time: str, end_time: str, frames: List[Dict], 
                       profile: Dict[str, Any]) -> Tuple[str, List[Dict]]:
        """
        Rescan a video segment at higher detail using Responses API.
        
        Args:
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            frames: High-fps frames for detailed analysis
            profile: Profile configuration
            
        Returns:
            Tuple of (detailed_narrative, event_log)
        """
        try:
            # Use rescan prompt from profile
            rescan_prompt = profile["rescan_prompt"].format(
                start_time=start_time, 
                end_time=end_time
            )
            
            # Instructions for detailed rescan
            instructions = "You are performing a detailed rescan of a video segment."
            
            # Add context about what we're rescanning
            input_text = f"""{rescan_prompt}

Previous analysis context: {self.context_state}

Analyze these frames with much higher detail, focusing on subtle movements and events that may have been missed in the initial scan.

{rescan_prompt}"""
            
            # Convert frames to base64
            base64_frames = self._frames_to_base64(frames)
            
            # Create input with multimodal content
            input_content = [
                {
                    "type": "input_text",
                    "text": input_text
                }
            ] + [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{frame_data}"
                } for frame_data in base64_frames
            ]
            
            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=[{
                    "role": "user",
                    "content": input_content
                }],
                previous_response_id=self.previous_response_id,  # Chain with previous context
                store=True
            )
            
            # Use the convenience property from Responses API
            if not response or not hasattr(response, 'output_text'):
                raise ValueError("Invalid API response: missing output_text")
            
            narrative_text = response.output_text
            
            if not narrative_text:
                raise ValueError("Invalid API response: empty output_text")
            
            # Update response ID
            self.previous_response_id = response.id
            
            # Ensure narrative text is properly encoded
            if isinstance(narrative_text, str):
                narrative_text = narrative_text.encode('utf-8', errors='replace').decode('utf-8')
            
            event_log = self._extract_events_from_response(narrative_text)
            
            return narrative_text, event_log
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error rescanning segment: {error_msg}")
            return f"Error during rescan: {error_msg}", []
    
    def condense_context(self, full_transcript: List[str]) -> str:
        """
        Condense full transcript into context state using GPT-5 Responses API.
        
        Args:
            full_transcript: List of narrative texts from all batches
            
        Returns:
            Condensed context state
        """
        try:
            # Join transcript with timestamps
            transcript_text = "\n\n".join(full_transcript)
            
            # Use context condensation instructions
            instructions = """You are maintaining a running summary of a video as it is being narrated. 
Your task is to take the current full transcript so far and compress it into a concise "state summary"."""
            
            # Input text for condensation
            input_text = f"""Condensed State Summary Guidelines: 
- Use no more than 150 words. 
- Preserve chronological flow. 
- Keep timestamps coarse (to the nearest ~10–15 seconds). 
- Do not include stylistic narration, just a factual condensed state. 
- Focus on continuity—what should be remembered for interpreting the next frames.
- Capture: Key events (with approximate timestamps), Main actors or objects of focus, Current scene status (who/what is present, what is happening), Any unresolved actions (e.g., "Person is reaching toward the door, action incomplete")

Transcript to condense:
{transcript_text}

Output format: [Condensed State Summary]"""
            
            # Make API call for condensation using Responses API
            response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=input_text,
                store=False  # Don't store condensation calls
            )
            
            # Use the convenience property from Responses API
            if not response or not hasattr(response, 'output_text'):
                raise ValueError("Invalid API response: missing output_text")
            
            condensed_state = response.output_text
            
            if not condensed_state:
                raise ValueError("Invalid API response: empty output_text")
            
            # Ensure condensed state is properly encoded
            if isinstance(condensed_state, str):
                condensed_state = condensed_state.encode('utf-8', errors='replace').decode('utf-8')
            
            return condensed_state
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error condensing context: {error_msg}")
            # Fallback to simple truncation
            return "Context condensation failed. Using fallback summary."
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript as markdown."""
        return "\n\n".join(self.full_transcript)
    
    def get_event_timeline(self) -> List[Dict]:
        """Get the complete event timeline."""
        # Filter to only include dict items (skip any string or non-dict items)
        dict_events = [e for e in self.event_log if isinstance(e, dict)]
        return sorted(dict_events, key=lambda x: x.get('timestamp', '00:00:00'))
    
    def assess_standardized_patient_encounter(self, narrative: str) -> str:
        """
        Create assessment report for standardized patient encounter using Responses API.
        
        Args:
            narrative: Narrative reconstruction from Pass 2
            
        Returns:
            JSON assessment report
        """
        try:
            # Assessment instructions
            instructions = "You are a medical faculty assessor evaluating a medical student's performance in a standardized patient encounter."
            
            assessment_input = f"""Your task is to evaluate a medical student's performance based on a written narrative reconstruction of the encounter.  
Do not critique the narrative. Judge the student's performance as if you had observed the actual encounter.  

Use the following rubric categories:
1. History Taking
2. Physical Examination
3. Communication Skills
4. Clinical Reasoning / Organization
5. Professionalism
6. Patient Education and Closure

CRITICAL GRADING STANDARDS
- Apply rigorous medical education standards typical of faculty assessors
- Score conservatively: reserve 5s for truly exceptional performance, use 3s for adequate but unremarkable work
- Default to mid-range scores (2-4); scores of 5 should be rare and reserved for outstanding demonstration
- Identify specific deficiencies and missed opportunities in each category
- Compare performance against expected competency for this stage of training

INSTRUCTIONS
- Base all judgments only on what is visible in the narrative (student speech and actions). Do not infer or imagine unstated behaviors.
- For each category, provide:
  • Critical score (1–5 scale: 1 = unsatisfactory, 2 = needs improvement, 3 = adequate, 4 = good, 5 = exceptional).  
  • Two to three sentences of CASE-SPECIFIC feedback citing exact observed moments from THIS encounter.
  • Identify specific gaps, missed questions, or areas needing improvement.
  • Reference the actual patient presentation and student's handling of THIS specific case.
- At the end, provide an overall comment (3-5 sentences) that:
  • Summarizes performance with specific references to THIS case
  • Identifies concrete areas for improvement based on what was observed
  • Notes any concerning patterns or missed critical elements
  • Provides actionable feedback tied to THIS specific encounter

OUTPUT FORMAT
{{
  "History Taking": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Physical Examination": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Communication Skills": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Clinical Reasoning": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Professionalism": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Patient Education and Closure": {{"score": X, "feedback": "Case-specific critique with exact examples from this encounter..."}},
  "Overall": "Case-specific summary analyzing THIS student's performance in THIS encounter, with concrete areas for improvement."
}}

RULES
- No speculation, no invented dialogue or actions.  
- Be critical but fair - identify both strengths AND weaknesses with specific examples.
- All feedback must reference THIS specific case and encounter.
- Emphasize alignment with rigorous medical education standards.
- Be more demanding than lenient - medical education requires high standards.

NARRATIVE TO ASSESS:
{narrative}"""

            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=assessment_input,
                store=True
            )
            
            # Use the convenience property from Responses API
            if not response or not hasattr(response, 'output_text'):
                raise ValueError("Invalid API response: missing output_text")
            
            assessment_text = response.output_text
            
            if not assessment_text:
                raise ValueError("Invalid API response: empty output_text")
            
            # Ensure proper encoding
            if isinstance(assessment_text, str):
                assessment_text = assessment_text.encode('utf-8', errors='replace').decode('utf-8')
            
            return assessment_text
        
        except APITimeoutError as e:
            error_msg = f"OpenAI API request timed out after {self.timeout}s. The service may be overloaded or experiencing issues."
            print(f"⏱️  Timeout error creating assessment: {error_msg}")
            return f"Error during assessment: {error_msg}"
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API. Check your internet connection."
            print(f"🔌 Connection error creating assessment: {error_msg}")
            return f"Error during assessment: {error_msg}"
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error creating assessment: {error_msg}")
            return f"Error during assessment: {error_msg}"

    def create_enhanced_narrative_chunked(self, transcript: str, events: List[Dict], 
                                        audio_transcript: str = "", yamnet_results: Dict = None,
                                        profile: Dict[str, Any] = None) -> str:
        """
        Create enhanced narrative using chunked processing for large transcripts.
        
        Args:
            transcript: Raw transcript from frame analysis
            events: Event timeline from frame analysis
            audio_transcript: Audio transcription if available
            yamnet_results: Audio classification results if available
            profile: Analysis profile configuration
            
        Returns:
            Enhanced coherent narrative
        """
        MAX_CHUNK_SIZE = 150000  # ~150KB per chunk to stay well under context limits
        
        try:
            # If transcript is small enough, use standard method
            if len(transcript) <= MAX_CHUNK_SIZE:
                return self.create_enhanced_narrative(transcript, events, audio_transcript, yamnet_results, profile)
            
            print(f"📊 Large transcript detected ({len(transcript):,} chars). Using chunked processing...")
            
            # Split transcript into chunks by finding natural break points (JSON array boundaries)
            chunks = self._split_transcript_into_chunks(transcript, MAX_CHUNK_SIZE)
            print(f"📦 Split into {len(chunks)} chunks")
            
            # Process each chunk into a sub-narrative (PARALLEL for Tier 4 speed)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_chunk(chunk_data):
                i, chunk = chunk_data
                print(f"✍️  Processing chunk {i+1}/{len(chunks)}...")
                
                instructions = """You are a clinical documentation specialist. Create a factual observational record from this portion of video analysis data."""
                
                input_text = f"""Create a clinical observational record for this segment of the video analysis:

FRAME-BY-FRAME TRANSCRIPT (Segment {i+1}/{len(chunks)}):
{chunk}

SYNTHESIS REQUIREMENTS:
- Write in flat, undramatic clinical language
- Document only what can be directly seen and heard
- Use objective, observational terminology
- Maintain strict chronological accuracy
- Avoid novelistic descriptions, dramatic embellishments, or inferred emotions
- Use clinical documentation style (factual, neutral, precise)
- This is part {i+1} of {len(chunks)}, so use consistent clinical tone

Output a factual observational record for just this segment."""
                
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    instructions=instructions,
                    input=input_text,
                    store=True
                )
                
                if response and hasattr(response, 'output_text'):
                    return (i, response.output_text)
                else:
                    raise ValueError(f"Invalid response for chunk {i+1}")
            
            # Process chunks in parallel (up to 10 concurrent for Tier 4)
            max_workers = min(10, len(chunks))
            print(f"🚀 Processing {len(chunks)} narrative chunks in parallel ({max_workers} workers)...")
            
            sub_narratives_dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_chunk, (i, chunk)): i for i, chunk in enumerate(chunks)}
                
                for future in as_completed(futures):
                    try:
                        i, narrative_text = future.result(timeout=180)
                        sub_narratives_dict[i] = narrative_text
                        print(f"✅ Chunk {i+1}/{len(chunks)} complete")
                    except Exception as e:
                        print(f"❌ Chunk processing failed: {e}")
                        raise
            
            # Reconstruct in order
            sub_narratives = [sub_narratives_dict[i] for i in sorted(sub_narratives_dict.keys())]
            
            # Now combine all sub-narratives into final coherent narrative
            print(f"🔗 Combining {len(sub_narratives)} narrative segments...")
            
            combined_input = f"""You are a clinical documentation specialist. Combine these sequential observational records into one complete clinical record.

OBSERVATIONAL RECORD SEGMENTS:
"""
            for i, sub_narr in enumerate(sub_narratives):
                combined_input += f"\n\n--- SEGMENT {i+1} ---\n{sub_narr}"
            
            if audio_transcript:
                combined_input += f"\n\nAUDIO TRANSCRIPT (for context):\n{audio_transcript[:5000]}..."  # Include sample
            
            combined_input += """

SYNTHESIS REQUIREMENTS:
- Merge these segments into one complete observational record
- Write in flat, undramatic clinical language
- Document only what can be directly seen and heard
- Remove redundancy while maintaining all factual observations
- Maintain strict chronological accuracy
- Use objective, clinical documentation style throughout
- Avoid novelistic descriptions, dramatic embellishments, or inferred emotions
- Use clinical documentation style (factual, neutral, precise)

Output the complete unified clinical observational record."""
            
            final_response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions="You are a clinical documentation specialist creating a unified observational record from multiple segments.",
                input=combined_input,
                store=True
            )
            
            if not final_response or not hasattr(final_response, 'output_text'):
                raise ValueError("Invalid final narrative response")
            
            enhanced_narrative = final_response.output_text
            
            if isinstance(enhanced_narrative, str):
                enhanced_narrative = enhanced_narrative.encode('utf-8', errors='replace').decode('utf-8')
            
            print(f"✅ Chunked narrative synthesis complete ({len(enhanced_narrative):,} chars)")
            return enhanced_narrative
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error in chunked narrative synthesis: {error_msg}")
            return f"Error during narrative enhancement: {error_msg}"
    
    def _split_transcript_into_chunks(self, transcript: str, max_size: int) -> List[str]:
        """Split transcript into chunks at natural boundaries (JSON array elements)."""
        chunks = []
        current_chunk = ""
        
        # Try to split by JSON entries (look for timecode patterns)
        lines = transcript.split('\n')
        
        for line in lines:
            # If adding this line would exceed max_size, start new chunk
            if len(current_chunk) + len(line) > max_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            current_chunk += line + '\n'
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def create_enhanced_narrative(self, transcript: str, events: List[Dict], 
                                audio_transcript: str = "", yamnet_results: Dict = None,
                                profile: Dict[str, Any] = None) -> str:
        """
        Create enhanced narrative using GPT-5 Responses API (Pass 2).
        Automatically uses chunked processing for large transcripts.
        
        Args:
            transcript: Raw transcript from frame analysis
            events: Event timeline from frame analysis
            audio_transcript: Audio transcription if available
            yamnet_results: Audio classification results if available
            profile: Analysis profile configuration
            
        Returns:
            Enhanced coherent narrative
        """
        # Check if we need chunked processing
        if len(transcript) > 150000:
            return self.create_enhanced_narrative_chunked(transcript, events, audio_transcript, yamnet_results, profile)
        
        try:
            # Instructions for narrative enhancement
            instructions = """You are a clinical documentation specialist. Your task is to create a factual, observational record from frame-by-frame video analysis data."""
            
            # Prepare input with all available data
            input_text = f"""Create a clinical observational record from the following video analysis data:

FRAME-BY-FRAME TRANSCRIPT:
{transcript}

EVENT TIMELINE:
{json.dumps(events, indent=2)}"""
            
            if audio_transcript:
                input_text += f"\n\nAUDIO TRANSCRIPT:\n{audio_transcript}"
            
            if yamnet_results:
                input_text += f"\n\nAUDIO CLASSIFICATION:\n{json.dumps(yamnet_results, indent=2)}"
            
            input_text += """

SYNTHESIS REQUIREMENTS:
- Write in flat, undramatic clinical language
- Document only what can be directly seen and heard in the data
- Use objective, observational terminology
- Maintain strict chronological accuracy
- Avoid novelistic descriptions, dramatic embellishments, or inferred emotions
- Use clinical documentation style (factual, neutral, precise)
- Do not interpret or speculate beyond documented observations

Output a complete observational record documenting exactly what was seen and heard in this video."""
            
            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=input_text,
                previous_response_id=self.previous_response_id,  # Chain with previous analysis
                store=True
            )
            
            # Use the convenience property from Responses API
            if not response or not hasattr(response, 'output_text'):
                raise ValueError("Invalid API response: missing output_text")
            
            enhanced_narrative = response.output_text
            
            if not enhanced_narrative:
                raise ValueError("Invalid API response: empty output_text")
            
            # Update response ID
            self.previous_response_id = response.id
            
            # Ensure proper encoding
            if isinstance(enhanced_narrative, str):
                enhanced_narrative = enhanced_narrative.encode('utf-8', errors='replace').decode('utf-8')
            
            return enhanced_narrative
        
        except APITimeoutError as e:
            error_msg = f"OpenAI API request timed out after {self.timeout}s. The service may be overloaded or experiencing issues."
            print(f"⏱️  Timeout error creating narrative: {error_msg}")
            return f"Error during narrative enhancement: {error_msg}"
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API. Check your internet connection."
            print(f"🔌 Connection error creating narrative: {error_msg}")
            return f"Error during narrative enhancement: {error_msg}"
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error creating enhanced narrative: {error_msg}")
            return f"Error during narrative enhancement: {error_msg}"