"""
Video processing module for frame extraction and video handling.
"""

import cv2
import ffmpeg
import numpy as np
from typing import List, Tuple, Optional, Union
import os
import tempfile
import requests
from PIL import Image
import io

class VideoProcessor:
    """Handles video processing, frame extraction, and metadata management."""
    
    def __init__(self):
        self.video_path: Optional[str] = None
        self.fps: float = 1.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        self.frame_metadata: List[dict] = []
        
    def load_video(self, source: Union[str, bytes], fps: float = 1.0) -> bool:
        """
        Load video from file path, URL, or bytes.
        
        Args:
            source: Video source (file path, URL, or bytes)
            fps: Frames per second to extract
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.fps = fps
            
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # Download from URL
                    response = requests.get(source)
                    response.raise_for_status()
                    video_bytes = response.content
                    self.video_path = self._save_temp_video(video_bytes)
                else:
                    # Local file - try multiple path variations if original doesn't exist
                    if os.path.exists(source):
                        self.video_path = source
                    else:
                        # Try temp_videos with temp_ prefix
                        basename = os.path.basename(source)
                        temp_path = os.path.join("temp_videos", f"temp_{basename}")
                        if os.path.exists(temp_path):
                            self.video_path = temp_path
                        else:
                            # Try temp_videos without temp_ prefix
                            temp_path2 = os.path.join("temp_videos", basename)
                            if os.path.exists(temp_path2):
                                self.video_path = temp_path2
                            else:
                                raise FileNotFoundError(f"Video file not found: {source} (also tried {temp_path}, {temp_path2})")
            else:
                # Bytes input
                self.video_path = self._save_temp_video(source)
            
            # Get video properties
            self._get_video_properties()
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def _save_temp_video(self, video_bytes: bytes) -> str:
        """Save video bytes to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_bytes)
        temp_file.close()
        return temp_file.name
    
    def _get_video_properties(self):
        """Extract video properties using ffmpeg."""
        try:
            probe = ffmpeg.probe(self.video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            self.duration = float(probe['format']['duration'])
            self.total_frames = int(video_info['nb_frames'])
            print(f"FFmpeg: Duration={self.duration}, Frames={self.total_frames}")
            
        except Exception as e:
            print(f"Error getting video properties with FFmpeg: {e}")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_orig = cap.get(cv2.CAP_PROP_FPS)
                self.duration = self.total_frames / fps_orig if fps_orig > 0 else 0
                print(f"OpenCV fallback: Duration={self.duration}, Frames={self.total_frames}, FPS={fps_orig}")
            else:
                print("Failed to open video with OpenCV")
                self.duration = 0
                self.total_frames = 0
            cap.release()
    
    def extract_frames(self, start_time: float = 0.0, end_time: Optional[float] = None, 
                      custom_fps: Optional[float] = None) -> List[dict]:
        """
        Extract frames from video with metadata using robust FFmpeg approach.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            custom_fps: Override default fps for this extraction
            
        Returns:
            List of frame metadata dictionaries
        """
        if not self.video_path:
            raise ValueError("No video loaded")
        
        if not self.duration or self.duration <= 0:
            raise ValueError(f"Invalid video duration: {self.duration}")
        
        fps = custom_fps if custom_fps else self.fps
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")
        
        end_time = end_time if end_time else max(0, self.duration - 1.0)  # Use 1 second before end as final frame
        
        print(f"Extracting frames: start={start_time}, end={end_time}, fps={fps}, duration={self.duration}")
        
        # Try FFmpeg extraction first (more robust)
        try:
            frames = self._extract_frames_ffmpeg(start_time, end_time, fps)
            if frames:
                print(f"Extracted {len(frames)} frames using FFmpeg")
                return frames
        except Exception as e:
            print(f"FFmpeg extraction failed: {e}, falling back to OpenCV")
        
        # Fallback to OpenCV with more resilient frame seeking
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError("Failed to open video for frame extraction")
        
        # Get total frame count and FPS for more accurate positioning
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} total frames at {video_fps} FPS")
        
        # Calculate frame intervals
        frame_interval = 1.0 / fps
        current_time = start_time
        consecutive_failures = 0
        max_failures = 10  # Allow some failures but not endless loop
        
        while current_time < end_time and consecutive_failures < max_failures:
            # Try frame-based positioning as alternative to time-based
            frame_number = int(current_time * video_fps)
            
            # Try both time-based and frame-based positioning
            success = False
            
            # Method 1: Time-based
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            
            if not ret and frame_number < total_frames:
                # Method 2: Frame-based
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create frame metadata
                frame_data = {
                    'frame_id': len(frames),
                    'timestamp': current_time,
                    'timestamp_formatted': self._format_timestamp(current_time),
                    'frame': frame_rgb,
                    'frame_pil': Image.fromarray(frame_rgb)
                }
                
                frames.append(frame_data)
                consecutive_failures = 0  # Reset failure counter
            else:
                consecutive_failures += 1
                print(f"Failed to read frame at time {current_time} (failure {consecutive_failures})")
            
            current_time += frame_interval
        
        cap.release()
        print(f"Extracted {len(frames)} frames")
        return frames
    
    def _extract_frames_ffmpeg(self, start_time: float, end_time: float, fps: float) -> List[dict]:
        """Extract frames using FFmpeg for better compatibility."""
        import tempfile
        import os
        import glob
        
        frames = []
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            output_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
            
            # Use FFmpeg to extract frames
            try:
                (
                    ffmpeg
                    .input(self.video_path, ss=start_time, t=(end_time - start_time))
                    .filter('fps', fps=fps)
                    .output(output_pattern)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                raise Exception(f"FFmpeg frame extraction failed: {e.stderr.decode()}")
            
            # Load extracted frames
            frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg")))
            
            for i, frame_file in enumerate(frame_files):
                timestamp = start_time + (i / fps)
                if timestamp >= end_time:
                    break
                
                # Load frame using PIL
                frame_pil = Image.open(frame_file)
                frame_rgb = np.array(frame_pil)
                
                frame_data = {
                    'frame_id': len(frames),
                    'timestamp': timestamp,
                    'timestamp_formatted': self._format_timestamp(timestamp),
                    'frame': frame_rgb,
                    'frame_pil': frame_pil
                }
                
                frames.append(frame_data)
        
        return frames
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_frame_at_time(self, timestamp: float) -> Optional[dict]:
        """Get a single frame at specific timestamp."""
        frames = self.extract_frames(start_time=timestamp, end_time=timestamp + 0.1, custom_fps=1)
        return frames[0] if frames else None
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.video_path and os.path.exists(self.video_path):
            try:
                os.unlink(self.video_path)
            except:
                pass
    
    def frames_to_base64(self, frames: List[dict]) -> List[str]:
        """
        Convert frames to base64 strings for API transmission.
        
        Args:
            frames: List of frame metadata dictionaries
            
        Returns:
            List of base64 encoded frame strings
        """
        import base64
        import io
        
        base64_frames = []
        for frame_data in frames:
            # Convert PIL image to base64
            img_buffer = io.BytesIO()
            frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            base64_frames.append(img_str)
        
        return base64_frames
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

class FrameBatchProcessor:
    """Handles batching of frames for API calls."""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
    
    def create_batches(self, frames: List[dict]) -> List[List[dict]]:
        """
        Create batches of frames for processing.
        
        Args:
            frames: List of frame metadata dictionaries
            
        Returns:
            List of frame batches
        """
        batches = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
