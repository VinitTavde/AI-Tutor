import os
import sys
import tempfile
import numpy as np
import torch
import soundfile as sf
import librosa
from typing import Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VibeVoiceService:
    """
    Integration service for VibeVoice podcast generation in AI_Tutor
    """
    
    def __init__(self, podcast_path: str = None):
        """
        Initialize VibeVoice service
        
        Args:
            podcast_path: Path to the podcast project directory
        """
        # Auto-detect podcast path or use default
        if podcast_path is None:
            # Try different possible paths
            possible_paths = [
                "f:/Kreative-Space/podcast",  # Local Windows
                "/content/podcast",           # Google Colab
                "./podcast",                  # Relative path
                "../podcast",                 # Parent directory
                os.path.join(os.getcwd(), "podcast")  # Current directory
            ]
            
            self.podcast_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.podcast_path = path
                    break
            
            if self.podcast_path is None:
                logger.warning("Podcast directory not found. VibeVoice service will be disabled.")
                self.podcast_path = "/tmp/podcast_fallback"  # Fallback path
        else:
            self.podcast_path = podcast_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_steps = 5
        
        # Add podcast path to sys.path for imports
        if self.podcast_path and os.path.exists(self.podcast_path) and self.podcast_path not in sys.path:
            sys.path.append(self.podcast_path)
        
        # Initialize VibeVoice components
        self.model = None
        self.processor = None
        self.available_voices = {}
        self.service_available = False
        
        try:
            self._load_vibevoice_components()
            self._setup_voice_presets()
            self.service_available = True
            logger.info("VibeVoice service initialized successfully")
        except Exception as e:
            logger.warning(f"VibeVoice service not available: {e}")
            self.service_available = False
    
    def _load_vibevoice_components(self):
        """Load VibeVoice model and processor"""
        original_cwd = os.getcwd()
        try:
            # Change to podcast directory for imports
            if self.podcast_path and os.path.exists(self.podcast_path):
                os.chdir(self.podcast_path)
                logger.info(f"Changed to podcast directory: {self.podcast_path}")
            
            from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from processor.vibevoice_processor import VibeVoiceProcessor
            
            model_path = "microsoft/VibeVoice-1.5B"
            
            logger.info(f"Loading VibeVoice processor from {model_path}")
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            logger.info("VibeVoice processor loaded successfully")
            
            logger.info(f"Loading VibeVoice model from {model_path}")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16
            )
            logger.info("VibeVoice model loaded successfully (on CPU initially)")
            
            # Keep on CPU initially, move to device only during generation
            # This matches the pattern in the original podcast app
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            logger.info(f"VibeVoice service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VibeVoice components: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.model = None
            self.processor = None
            raise  # Re-raise to trigger fallback to mock service
        finally:
            # Always change back to original directory
            os.chdir(original_cwd)
    
    def _setup_voice_presets(self):
        """Setup available voice presets from the voices directory"""
        voices_dir = os.path.join(self.podcast_path, "voices")
        
        if not os.path.exists(voices_dir):
            logger.warning(f"Voices directory not found at {voices_dir}")
            return
        
        audio_files = [f for f in os.listdir(voices_dir)
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
        
        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            self.available_voices[name] = os.path.join(voices_dir, audio_file)
        
        logger.info(f"Loaded {len(self.available_voices)} voice presets: {list(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file"""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            logger.error(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def get_default_speakers(self) -> Tuple[str, str]:
        """Get default speaker voices for two-speaker podcast"""
        # Prefer English speakers for default
        english_voices = [v for v in self.available_voices.keys() if v.startswith('en-')]
        
        if len(english_voices) >= 2:
            # Get one male and one female voice if available
            male_voices = [v for v in english_voices if 'man' in v or 'male' in v.lower()]
            female_voices = [v for v in english_voices if 'woman' in v or 'female' in v.lower()]
            
            speaker1 = female_voices[0] if female_voices else english_voices[0]
            speaker2 = male_voices[0] if male_voices else english_voices[1]
        else:
            # Fallback to any available voices
            voices = list(self.available_voices.keys())
            speaker1 = voices[0] if len(voices) > 0 else None
            speaker2 = voices[1] if len(voices) > 1 else speaker1
        
        return speaker1, speaker2
    
    @torch.inference_mode()
    def generate_voice_podcast(self, script: str, speaker1: str = None, speaker2: str = None, 
                             cfg_scale: float = 1.3) -> Tuple[Optional[str], str]:
        """
        Generate voice podcast from script
        
        Args:
            script: Podcast script with Speaker 1: and Speaker 2: format
            speaker1: Voice preset name for Speaker 1
            speaker2: Voice preset name for Speaker 2
            cfg_scale: CFG scale for generation
            
        Returns:
            Tuple of (audio_file_path, log_message)
        """
        if not self.model or not self.processor:
            return None, "❌ VibeVoice model not available. Please check installation."
        
        try:
            # Move model to device for generation
            logger.info(f"Moving VibeVoice model to {self.device} for generation")
            self.model = self.model.to(self.device)
            
            # Use default speakers if not specified
            if not speaker1 or not speaker2:
                default_speaker1, default_speaker2 = self.get_default_speakers()
                speaker1 = speaker1 or default_speaker1
                speaker2 = speaker2 or default_speaker2
            
            if not speaker1 or not speaker2:
                return None, "❌ No voice presets available. Please check voices directory."
            
            # Check if speakers exist
            if speaker1 not in self.available_voices or speaker2 not in self.available_voices:
                return None, f"❌ Speaker voices not found: {speaker1}, {speaker2}"
            
            log = f"🎙️ Generating voice podcast with speakers: {speaker1}, {speaker2}\n"
            
            # Load voice samples
            voice1_path = self.available_voices[speaker1]
            voice2_path = self.available_voices[speaker2]
            
            voice1_audio = self.read_audio(voice1_path)
            voice2_audio = self.read_audio(voice2_path)
            
            if len(voice1_audio) == 0 or len(voice2_audio) == 0:
                return None, "❌ Failed to load voice samples"
            
            voice_samples = [voice1_audio, voice2_audio]
            log += f"✅ Loaded voice samples successfully\n"
            
            # Format script for VibeVoice (ensure proper Speaker X: format)
            lines = script.strip().split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Speaker 1:') or line.startswith('Speaker 2:'):
                    formatted_lines.append(line)
                else:
                    # Auto-assign speaker if not specified
                    speaker_id = (len(formatted_lines) % 2) + 1
                    formatted_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_lines)
            log += f"📝 Formatted script with {len(formatted_lines)} dialogue turns\n"
            
            # Process with VibeVoice
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            log += "🔄 Processing with VibeVoice...\n"
            
            # Generate audio
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                show_progress_bar=False,
            )
            
            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]
                audio = audio_tensor.cpu().float().numpy()
            else:
                return None, "❌ No audio generated by the model"
            
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            # Save audio file
            sample_rate = 24000
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"voice_podcast_{timestamp}.wav")
            sf.write(file_path, audio, sample_rate)
            
            duration = len(audio) / sample_rate
            log += f"✅ Voice podcast generated successfully\n"
            log += f"🎵 Duration: {duration:.2f} seconds\n"
            log += f"💾 Saved to: {file_path}\n"
            
            return file_path, log
            
        except Exception as e:
            logger.error(f"Error generating voice podcast: {str(e)}")
            return None, f"❌ Voice generation error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if VibeVoice service is available"""
        return self.service_available and self.model is not None and self.processor is not None
