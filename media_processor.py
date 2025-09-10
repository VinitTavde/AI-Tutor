import os
import logging
from typing import Optional, Dict
import yt_dlp
import tempfile

logger = logging.getLogger(__name__)

class MediaProcessor:
    def __init__(self):
        """Initialize the media processor with yt-dlp."""
        self.temp_dir = tempfile.gettempdir()
        
    def _get_ydl_opts(self, output_template: str, max_filesize_mb: int = 500) -> Dict:
        """Get the options for yt-dlp."""
        return {
            'format': 'bestaudio/best',  
            'outtmpl': output_template, 
            'noplaylist': True,         
            'quiet': True,              
            'no_warnings': True,
            'extractaudio': True,       
            'audioformat': 'wav',      
            'max_filesize': max_filesize_mb * 1024 * 1024,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
            }],
        }

    def get_video_info(self, url: str) -> Optional[Dict]:
        """
        Get video information without downloading.
        
        Returns:
            A dictionary with video info (title, duration, etc.) or None if invalid.
        """
        try:
            ydl_opts = {'quiet': True, 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get('title', 'No Title'),
                    "duration": info.get('duration', 0),
                    "uploader": info.get('uploader', 'N/A'),
                    "thumbnail": info.get('thumbnail', ''),
                    "webpage_url": info.get('webpage_url', url)
                }
        except Exception as e:
            logger.error(f"Failed to get video info for {url}: {e}")
            return None

    def download_audio(self, url: str) -> Optional[str]:
        """
        Download audio from a URL and save it as a temporary wav file.
        
        Args:
            url: The URL to download from.
            
        Returns:
            The file path to the downloaded temporary audio file, or None if failed.
        """
        try:
            # Create a unique temporary filename
            temp_file_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            output_template = os.path.splitext(temp_file_path)[0] + '.%(ext)s'

            # Get yt-dlp options
            ydl_opts = self._get_ydl_opts(output_template)

            logger.info(f"Downloading audio from {url}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download([url])
                if error_code != 0:
                    logger.error(f"yt-dlp failed to download {url} with error code {error_code}")
                    return None
            
            # The output file should have been renamed to .wav by the postprocessor
            expected_wav_path = os.path.splitext(temp_file_path)[0] + '.wav'

            if os.path.exists(expected_wav_path):
                logger.info(f"Successfully downloaded and converted audio to {expected_wav_path}")
                return expected_wav_path
            else:
                logger.error(f"Downloaded file not found at expected path: {expected_wav_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download audio from {url}: {str(e)}")
            return None 