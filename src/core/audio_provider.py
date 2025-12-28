"""
Audio Provider abstraction layer for TTS/ASR services.
Provides a unified interface for different text-to-speech and speech-to-text providers.
"""

from typing import Optional, Dict, Any, List, Protocol
from abc import ABC, abstractmethod
import asyncio
import tempfile
import os
from pathlib import Path
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal, QUrl, QTimer

from ..core.logging_config import get_logger

# Qt multimedia imports for audio playback
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    QT_MULTIMEDIA_AVAILABLE = True
except ImportError:
    QT_MULTIMEDIA_AVAILABLE = False
    logger.warning("Qt Multimedia not available. TTS playback will not work.")

logger = get_logger(__name__)


class TTSProvider(Protocol):
    """Protocol for Text-to-Speech providers."""

    def speak(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech and return audio data.

        Args:
            text: Text to convert to speech
            **kwargs: Provider-specific options

        Returns:
            Audio data as bytes
        """
        ...

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices.

        Returns:
            List of voice dictionaries with 'name', 'language', etc.
        """
        ...

    def set_voice(self, voice_name: str):
        """
        Set the current voice.

        Args:
            voice_name: Name of the voice to use
        """
        ...


class ASRProvider(Protocol):
    """Protocol for Automatic Speech Recognition providers."""

    def listen(self, timeout: Optional[float] = None, **kwargs) -> str:
        """
        Listen for speech and convert to text.

        Args:
            timeout: Maximum time to listen in seconds
            **kwargs: Provider-specific options

        Returns:
            Transcribed text
        """
        ...

    def is_listening(self) -> bool:
        """
        Check if currently listening for speech.

        Returns:
            True if listening, False otherwise
        """
        ...


class AudioProvider(QObject):
    """
    Unified audio provider that handles both TTS and ASR.
    Supports multiple providers with a common interface.
    Implements QMediaPlayer control for playback.
    """
    
    # Playback signals
    playback_state_changed = pyqtSignal(str) # 'playing', 'paused', 'stopped'
    position_changed = pyqtSignal(int)       # position in milliseconds
    duration_changed = pyqtSignal(int)       # duration in milliseconds
    error_occurred = pyqtSignal(str)         # error message

    def __init__(self, openai_api_key: Optional[str] = None):
        super().__init__()
        self.tts_provider: Optional[TTSProvider] = None
        self.asr_provider: Optional[ASRProvider] = None
        self.openai_api_key = openai_api_key

        # Create temp directory with error handling and fallback
        self.temp_dir = self._create_temp_directory()

        # Audio settings
        self.tts_enabled = True
        self.asr_enabled = True
        self.tts_voice = "alloy"  # Default OpenAI voice
        self.tts_speed = 1.0
        self.asr_language = "en-US"
        self.audio_format = "mp3"

        # Media Player (Single Instance)
        self._player: Optional[QMediaPlayer] = None
        self._audio_output: Optional[QAudioOutput] = None
        self._init_player()

        # Initialize default providers
        self._init_providers()

    def _init_player(self):
        """Initialize the single QMediaPlayer instance."""
        if not QT_MULTIMEDIA_AVAILABLE:
            return

        try:
            self._player = QMediaPlayer()
            self._audio_output = QAudioOutput()
            self._player.setAudioOutput(self._audio_output)
            
            # Set default volume to 100%
            self._audio_output.setVolume(1.0)
            
            # Connect signals
            self._player.positionChanged.connect(self.position_changed.emit)
            self._player.durationChanged.connect(self.duration_changed.emit)
            self._player.playbackStateChanged.connect(self._on_playback_state_changed)
            self._player.errorOccurred.connect(self._on_player_error)
            
            logger.info("Audio player initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio player: {e}")

    def _create_temp_directory(self) -> Path:
        """Create temporary directory with error handling and fallback."""
        try:
            # Primary location: system temp directory
            primary_temp_dir = Path(tempfile.gettempdir()) / "personal_assistant_audio"
            primary_temp_dir.mkdir(exist_ok=True)
            logger.info(f"Created temp directory: {primary_temp_dir}")
            return primary_temp_dir
        except Exception as e:
            logger.warning(f"Failed to create temp directory in system temp: {e}")
            try:
                # Fallback: current working directory
                fallback_temp_dir = Path.cwd() / "temp_audio"
                fallback_temp_dir.mkdir(exist_ok=True)
                logger.info(f"Using fallback temp directory: {fallback_temp_dir}")
                return fallback_temp_dir
            except Exception as e2:
                logger.error(f"Failed to create fallback temp directory: {e2}")
                # Last resort: memory-based temp (no actual directory)
                logger.warning("Using memory-only mode - temp files will not be cleaned up")
                # Create a dummy path that won't be used in file operations
                return Path("dummy_temp_path")

    def _init_providers(self):
        """Initialize default TTS and ASR providers."""
        try:
            # Try to initialize OpenAI TTS provider
            self.tts_provider = OpenAITTSProvider(api_key=self.openai_api_key)
            logger.info("OpenAI TTS provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI TTS provider: {e}")
            self.tts_provider = None

        try:
            # Try to initialize speech recognition provider
            self.asr_provider = SpeechRecognitionASRProvider()
            logger.info("Speech Recognition ASR provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ASR provider: {e}")
            self.asr_provider = None

    def _on_playback_state_changed(self, state):
        """Handle player state changes."""
        state_str = "stopped"
        if state == QMediaPlayer.PlaybackState.PlayingState:
            state_str = "playing"
        elif state == QMediaPlayer.PlaybackState.PausedState:
            state_str = "paused"
        
        self.playback_state_changed.emit(state_str)
        logger.debug(f"Playback state changed: {state_str}")

    def _on_player_error(self, error, error_string):
        """Handle player errors."""
        msg = f"Player error: {error_string}"
        logger.error(msg)
        self.error_occurred.emit(msg)

    # Audio Control Methods

    def play_audio_file(self, file_path: str):
        """Play an audio file."""
        if not self._player:
            logger.warning("Player not initialized")
            return

        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Audio file not found: {file_path}")
                return

            self._player.setSource(QUrl(path.as_uri()))
            # Ensure volume is up
            if self._audio_output:
                self._audio_output.setVolume(1.0)
            
            self._player.play()
            logger.info(f"Playing audio: {file_path}")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    def pause_audio(self):
        """Pause playback."""
        if self._player:
            self._player.pause()

    def resume_audio(self):
        """Resume playback."""
        if self._player:
            self._player.play()

    def stop_audio(self):
        """Stop playback."""
        if self._player:
            self._player.stop()

    def seek_audio(self, offset_ms: int):
        """Seek forward/backward by offset in milliseconds."""
        if self._player and self._player.isSeekable():
            new_pos = self._player.position() + offset_ms
            new_pos = max(0, min(new_pos, self._player.duration()))
            self._player.setPosition(new_pos)

    def set_position(self, position_ms: int):
        """Set absolute playback position."""
        if self._player and self._player.isSeekable():
            self._player.setPosition(position_ms)

    # TTS Methods

    def speak_text(self, text: str, save_to_file: bool = False) -> Optional[str]:
        """
        Convert text to speech and optionally save to file.

        Args:
            text: Text to convert to speech
            save_to_file: Whether to save audio to a temporary file

        Returns:
            Path to audio file if save_to_file is True, None otherwise
        """
        if not self.tts_enabled or not self.tts_provider:
            logger.warning("TTS is disabled or no provider available")
            return None

        try:
            # Generate speech
            audio_data = self.tts_provider.speak(
                text,
                voice=self.tts_voice,
                speed=self.tts_speed
            )

            if save_to_file:
                # Save to temporary file
                timestamp = int(time.time())
                filename = f"tts_{timestamp}.{self.audio_format}"
                filepath = self.temp_dir / filename

                # Ensure directory exists before writing
                self.temp_dir.mkdir(parents=True, exist_ok=True)

                with open(filepath, 'wb') as f:
                    f.write(audio_data)

                logger.info(f"TTS audio saved to: {filepath}")
                return str(filepath)

            return None # Immediate playback (without save) removed in favor of worker model

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available TTS voices."""
        if not self.tts_provider:
            return []
        try:
            return self.tts_provider.get_available_voices()
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []

    def set_tts_voice(self, voice_name: str):
        """Set the TTS voice."""
        self.tts_voice = voice_name
        if self.tts_provider:
            try:
                self.tts_provider.set_voice(voice_name)
            except Exception as e:
                logger.error(f"Failed to set TTS voice: {e}")

    def set_tts_speed(self, speed: float):
        """Set the TTS speaking speed."""
        self.tts_speed = max(0.5, min(2.0, speed))

    # ASR Methods

    def listen_for_speech(self, timeout: Optional[float] = 10.0) -> Optional[str]:
        """Listen for speech and convert to text."""
        if not self.asr_enabled or not self.asr_provider:
            logger.warning("ASR is disabled or no provider available")
            return None

        try:
            text = self.asr_provider.listen(timeout=timeout, language=self.asr_language)
            logger.info(f"ASR transcription: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"ASR failed: {e}")
            return None

    def is_asr_listening(self) -> bool:
        """Check if ASR is currently listening."""
        if not self.asr_provider:
            return False
        try:
            return self.asr_provider.is_listening()
        except Exception as e:
            logger.error(f"Failed to check ASR listening status: {e}")
            return False

    # Configuration Methods

    def configure_tts(self, enabled: bool = True, voice: str = "alloy", speed: float = 1.0):
        """Configure TTS settings."""
        self.tts_enabled = enabled
        self.tts_voice = voice
        self.tts_speed = speed

    def configure_asr(self, enabled: bool = True, language: str = "en-US"):
        """Configure ASR settings."""
        self.asr_enabled = enabled
        self.asr_language = language

    def get_audio_settings(self) -> Dict[str, Any]:
        """Get current audio settings."""
        return {
            'tts_enabled': self.tts_enabled,
            'tts_voice': self.tts_voice,
            'tts_speed': self.tts_speed,
            'asr_enabled': self.asr_enabled,
            'asr_language': self.asr_language,
            'audio_format': self.audio_format
        }

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary audio files."""
        try:
            if not self.temp_dir.exists():
                return
                
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0

            for file_path in self.temp_dir.glob("*.mp3"):
                try:
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
                except Exception:
                    pass

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old audio files")

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        # Note: We avoid aggressive directory deletion here to prevent race conditions
        # when reloading modules. We only clean individual old files.
        try:
            if self._player:
                self._player.stop()
        except:
            pass


# Provider Implementations (Same as before)

class OpenAITTSProvider:
    """OpenAI TTS provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.voice = "alloy"
        self.speed = 1.0

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Provide api_key parameter or set OPENAI_API_KEY environment variable.")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    def speak(self, text: str, **kwargs) -> bytes:
        """Convert text to speech using OpenAI TTS."""
        voice = kwargs.get('voice', self.voice)
        speed = kwargs.get('speed', self.speed)

        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed
        )

        return b''.join(response.iter_bytes())

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available OpenAI voices."""
        return [
            {"name": "alloy", "language": "en", "gender": "neutral", "description": "Balanced and clear"},
            {"name": "echo", "language": "en", "gender": "male", "description": "Warm and clear"},
            {"name": "fable", "language": "en", "gender": "female", "description": "Expressive and warm"},
            {"name": "onyx", "language": "en", "gender": "male", "description": "Deep and authoritative"},
            {"name": "nova", "language": "en", "gender": "female", "description": "Youthful and energetic"},
            {"name": "shimmer", "language": "en", "gender": "female", "description": "Bright and clear"}
        ]

    def set_voice(self, voice_name: str):
        """Set the current voice."""
        available_voices = [v["name"] for v in self.get_available_voices()]
        if voice_name in available_voices:
            self.voice = voice_name
        else:
            logger.warning(f"Voice '{voice_name}' not available, using default 'alloy'")


class SpeechRecognitionASRProvider:
    """Speech Recognition ASR provider using system speech recognition."""

    def __init__(self):
        self.language = "en-US"
        self.is_currently_listening = False

        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except ImportError:
            raise ImportError("speech_recognition package not installed. Install with: pip install SpeechRecognition")

    def listen(self, timeout: Optional[float] = None, **kwargs) -> str:
        """Listen for speech and convert to text."""
        language = kwargs.get('language', self.language)

        try:
            self.is_currently_listening = True

            with self.microphone as source:
                logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            logger.info("Processing speech...")
            text = self.recognizer.recognize_google(audio, language=language)

            return text.strip()

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            raise
        finally:
            self.is_currently_listening = False

    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self.is_currently_listening
