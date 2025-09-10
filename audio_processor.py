import os
import logging
from typing import Optional, Tuple
import tempfile
import subprocess

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        """Initialize audio processor with local Whisper (free)."""
        self.max_file_size_mb = 500  # More generous limit for local processing
        self.supported_formats = {
            '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma',
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', 
            '.mpeg', '.mpg', '.3gp', '.3g2', '.f4v', '.asf', '.rm', '.rmvb'
        }
        self.speed_multiplier = 2.0  # 2x speed for faster transcription
        
        # Check if FFmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        
        # Try to import local Whisper
        try:
            import whisper
            self.whisper = whisper
            self.model = None  # Load on demand
            logger.info("Local Whisper available for audio processing")
        except ImportError:
            self.whisper = None
            logger.error("Whisper not installed. Install with: pip install openai-whisper")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in the system."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg available for audio processing")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Audio speed optimization disabled.")
            return False

    def _speed_up_audio(self, input_path: str, output_path: str, speed: float = 2.0) -> bool:
        """
        Speed up audio file using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output sped-up audio file
            speed: Speed multiplier (2.0 = 2x speed)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ffmpeg_available:
            return False
            
        try:
            # FFmpeg command to speed up audio
            # Using atempo filter (max 2.0x per filter, chain if needed)
            if speed <= 2.0:
                filter_chain = f"atempo={speed}"
            else:
                # For speed > 2x, chain multiple atempo filters
                num_filters = int(speed / 2.0)
                remaining_speed = speed / (2.0 ** num_filters)
                filter_chain = ",".join([f"atempo=2.0"] * num_filters)
                if remaining_speed > 1.0:
                    filter_chain += f",atempo={remaining_speed}"
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-filter:a', filter_chain,
                '-y',  # Overwrite output file
                output_path
            ]
            
            logger.info(f"Speeding up audio by {speed}x...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio successfully sped up to {speed}x")
                return True
            else:
                logger.warning(f"FFmpeg failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to speed up audio: {e}")
            return False

    def is_audio_file(self, filename: str) -> bool:
        """Check if the file is a supported audio format."""
        if not filename:
            return False
        
        file_extension = os.path.splitext(filename.lower())[1]
        return file_extension in self.supported_formats

    def validate_audio_file(self, file_path: str, filename: str) -> Tuple[bool, str]:
        """Validate audio file format and size."""
        # Check format
        if not self.is_audio_file(filename):
            return False, f"Unsupported audio format. Supported formats: {', '.join(self.supported_formats)}"
        
        # Check file size
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"Audio file too large. Maximum size: {self.max_file_size_mb}MB"
        except OSError:
            return False, "Could not read audio file"
        
        return True, "Audio file is valid"

    def transcribe_with_whisper(self, file_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio using local Whisper model (free) with speed optimization."""
        try:
            # Load model on demand (base is good balance of speed/accuracy)
            if self.model is None:
                logger.info("Loading Whisper model...")
                self.model = self.whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            
            # Try to speed up audio for faster transcription
            audio_to_transcribe = file_path
            temp_sped_file = None
            
            if self.ffmpeg_available:
                try:
                    # Create temporary file for sped-up audio
                    temp_sped_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_sped_file.close()
                    
                    # Speed up the audio
                    if self._speed_up_audio(file_path, temp_sped_file.name, self.speed_multiplier):
                        audio_to_transcribe = temp_sped_file.name
                        logger.info(f"Using {self.speed_multiplier}x sped-up audio for faster transcription")
                    else:
                        logger.info("Using original audio file (speed-up failed)")
                        
                except Exception as e:
                    logger.warning(f"Audio speed-up failed, using original: {e}")
            
            # Prepare transcription parameters
            kwargs = {"fp16": False}  # More compatible across systems
            if language and language != "auto":
                kwargs["language"] = language
            
            logger.info(f"Starting transcription...")
            
            # Transcribe
            result = self.model.transcribe(audio_to_transcribe, **kwargs)
            
            # Clean up temporary sped-up file
            if temp_sped_file and os.path.exists(temp_sped_file.name):
                try:
                    os.unlink(temp_sped_file.name)
                except:
                    pass  # Ignore cleanup errors
            
            logger.info("Transcription completed successfully")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {str(e)}")
            raise

    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Tuple[str, dict]:
        """
        Transcribe audio file to text using local Whisper (free) with speed optimization.
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr') or 'auto' for auto-detection
            
        Returns:
            Tuple of (transcribed_text, metadata_dict)
        """
        # Check if Whisper is available
        if not self.whisper:
            raise RuntimeError("Whisper not installed. Please install with: pip install openai-whisper")
        
        # Validate file first
        is_valid, validation_message = self.validate_audio_file(file_path, os.path.basename(file_path))
        if not is_valid:
            raise ValueError(validation_message)
        
        logger.info(f"Starting transcription of {file_path}")
        
        # Get file info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Transcribe using local Whisper (with speed optimization)
        transcribed_text = self.transcribe_with_whisper(file_path, language)
        
        # Check if transcription was successful
        if not transcribed_text:
            raise RuntimeError("Audio transcription failed. Please check your audio file quality.")
        
        # Prepare metadata
        metadata = {
            "transcription_method": "local_whisper",
            "speed_optimization": "2x" if self.ffmpeg_available else "disabled",
            "original_format": file_extension,
            "file_size_mb": round(file_size_mb, 2),
            "language": language or "auto",
            "word_count": len(transcribed_text.split())
        }
        
        logger.info(f"Transcription completed. Text length: {len(transcribed_text)} characters")
        return transcribed_text, metadata

    def get_supported_languages(self) -> dict:
        """Return supported language codes and names."""
        return {
            "auto": "Auto-detect",
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "no": "Norwegian",
            "fi": "Finnish",
            "pl": "Polish",
            "tr": "Turkish",
            "cs": "Czech",
            "hu": "Hungarian",
            "ro": "Romanian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sk": "Slovak",
            "sl": "Slovenian",
            "et": "Estonian",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "uk": "Ukrainian",
            "be": "Belarusian",
            "mk": "Macedonian",
            "mt": "Maltese",
            "ga": "Irish",
            "cy": "Welsh",
            "is": "Icelandic",
            "eu": "Basque",
            "ca": "Catalan",
            "gl": "Galician",
            "he": "Hebrew",
            "fa": "Persian",
            "ur": "Urdu",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "ml": "Malayalam",
            "kn": "Kannada",
            "gu": "Gujarati",
            "pa": "Punjabi",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Filipino"
        }