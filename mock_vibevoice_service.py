"""
Mock VibeVoice Service for Testing
When the full VibeVoice model is not available, this provides a fallback
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class MockVibeVoiceService:
    """
    Mock VibeVoice service that generates synthetic audio for testing
    """
    
    def __init__(self):
        """Initialize mock service"""
        self.service_available = True
        self.available_voices = {
            "en-Alice_woman": "Mock female voice",
            "en-Carter_man": "Mock male voice"
        }
        logger.info("Mock VibeVoice service initialized")
    
    def get_default_speakers(self) -> Tuple[str, str]:
        """Get default speaker voices"""
        return "en-Alice_woman", "en-Carter_man"
    
    def generate_voice_podcast(self, script: str, speaker1: str = None, speaker2: str = None, 
                             cfg_scale: float = 1.3) -> Tuple[Optional[str], str]:
        """
        Generate mock voice podcast (returns path to generated file)
        
        Args:
            script: Podcast script
            speaker1: Voice preset for Speaker 1
            speaker2: Voice preset for Speaker 2
            cfg_scale: Generation parameter
            
        Returns:
            Tuple of (audio_file_path, log_message)
        """
        
        # Use default speakers if not specified
        if not speaker1 or not speaker2:
            speaker1, speaker2 = self.get_default_speakers()
        
        try:
            # Generate a simple synthetic audio (beep tones)
            sample_rate = 24000
            duration = min(len(script) * 0.1, 30)  # Max 30 seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create alternating tones for two speakers
            freq1 = 440  # A4 note for speaker 1
            freq2 = 523  # C5 note for speaker 2
            
            # Simple alternating pattern
            audio = np.zeros_like(t)
            segment_length = len(t) // 4
            
            for i in range(4):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i < 3 else len(t)
                
                if i % 2 == 0:
                    # Speaker 1 segments
                    audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq1 * t[start_idx:end_idx])
                else:
                    # Speaker 2 segments
                    audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq2 * t[start_idx:end_idx])
            
            # Apply envelope to make it sound more natural
            envelope = np.exp(-t * 0.5)  # Decay envelope
            audio *= envelope
            
            # Save to outputs directory
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"mock_voice_podcast_{timestamp}.wav")
            
            # Save using soundfile if available, otherwise skip saving
            try:
                import soundfile as sf
                sf.write(file_path, audio, sample_rate)
                save_info = f"💾 Saved to: {file_path}\n"
            except ImportError:
                save_info = "💾 Audio generated (soundfile not available for saving)\n"
                file_path = None
            
            log = f"""🎙️ Mock Voice Podcast Generated Successfully!
📝 Script length: {len(script)} characters
🎭 Speakers: {speaker1} (tone: {freq1}Hz), {speaker2} (tone: {freq2}Hz)
⏱️ Duration: {duration:.2f} seconds
🎵 Sample rate: {sample_rate}Hz
{save_info}
⚠️ Note: This is a mock audio for testing. Install VibeVoice for real speech synthesis."""
            
            return file_path, log
            
        except Exception as e:
            logger.error(f"Error generating mock voice podcast: {str(e)}")
            return None, f"❌ Mock voice generation error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if mock service is available"""
        return self.service_available
