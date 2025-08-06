"""Multi-channel audio playback utility for surround sound."""

import asyncio
import pyaudio
import numpy as np
from typing import Optional, Union, Dict, Any

from src.utils.logging import setup_logger
from src.utils.config import load_config
from src.utils.audio_effects import AudioEffects

logger = setup_logger("multi_channel_player")

class MultiChannelPlayer:
    """Audio player for multi-channel (surround sound) output."""
    
    def __init__(self):
        """Initialize the multi-channel audio player."""
        self.main_config = load_config("main")
        self.py_audio = pyaudio.PyAudio()
        self.audio_effects = AudioEffects()  # Initialize audio effects processor
        
        # Handle device configuration
        device_setting = self.main_config["audio"]["device_index"]
        if isinstance(device_setting, str):
            if device_setting.lower() == "default":
                self.device_index = None
            else:
                # For string device names like "plughw:CARD=U5,DEV=0", keep as string
                # PyAudio will handle it properly
                self.device_index = device_setting
        else:
            self.device_index = device_setting
        
        self.sample_rate = self.main_config["audio"]["sample_rate"]
        self.channels = self.main_config["audio"]["channels"]
        
        # Try to initialize with requested channels, fallback if needed
        self.stream = None
        self._initialize_stream()
        
        logger.info(f"Initialized multi-channel player with {self.channels} channels on device {self.device_index}")
    
    def _initialize_stream(self):
        """Initialize the audio stream with fallback for unsupported channel counts."""
        try:
            self.stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_index if isinstance(self.device_index, int) else None
            )
        except OSError as e:
            if "Invalid number of channels" in str(e):
                logger.warning(f"Device doesn't support {self.channels} channels, falling back to stereo")
                self.channels = 2  # Fallback to stereo
                self.stream = self.py_audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.device_index if isinstance(self.device_index, int) else None
                )
            else:
                logger.error(f"Error starting audio stream: {e}")
                raise
    
    async def start(self):
        """Start the audio player (async compatibility)."""
        # Stream is already initialized in __init__, so this is just for API compatibility
        logger.info("Multi-channel player started")
    
    async def stop(self):
        """Stop the audio player (async compatibility)."""
        self.cleanup()
        logger.info("Multi-channel player stopped")
    
    def apply_audio_effects(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        effects_config: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply audio effects to the audio data if effects are configured.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate in Hz
            effects_config: Optional effects configuration dictionary
            
        Returns:
            Processed audio data
        """
        if effects_config and len(effects_config) > 0:
            return self.audio_effects.apply_effects(audio_data, sample_rate, effects_config)
        return audio_data
    
    def play_on_channel(self, audio_data: np.ndarray, sample_rate: int, channel: int, effects_config: Optional[Dict[str, Any]] = None):
        """Play audio data on a specific channel with optional effects."""
        if self.stream is None:
            logger.error("Audio stream not initialized")
            return
        
        # Apply audio effects if configured
        processed_audio = self.apply_audio_effects(audio_data, sample_rate, effects_config)
        
        # Create multi-channel audio buffer
        audio_buffer = np.zeros((len(processed_audio), self.channels), dtype=np.int16)
        
        # Place audio on specified channel (with bounds checking)
        target_channel = min(channel, self.channels - 1)
        if channel >= self.channels:
            logger.warning(f"Requested channel {channel} not available, using channel {target_channel}")
        
        audio_buffer[:, target_channel] = processed_audio
        
        # Convert to bytes and play
        audio_bytes = audio_buffer.tobytes()
        self.stream.write(audio_bytes)
        
        logger.info(f"Played audio on channel {target_channel} of {self.channels} channels")
    
    def play_to_position_sync(self, audio_data: np.ndarray, position: str, sample_rate: int, effects_config: Optional[Dict[str, Any]] = None):
        """Play audio data to a specific position (synchronous method) with optional effects."""
        # Map position to channel based on ASUS Xonar U5 actual tested layout
        position_to_channel = {
            "front-left": 0,      # Confirmed: Channel 0 -> Front-left
            "front-right": 1,     # Confirmed: Channel 1 -> Front-right  
            "center": 2,          # No center speaker, map to rear-left
            "rear-left": 2,       # Confirmed: Channel 2 -> Rear-left
            "rear-right": 3,      # Confirmed: Channel 3 -> Rear-right
            "lfe": 5,            # Confirmed: Channel 5 -> LFE/Subwoofer
            "subwoofer": 5,      # Same as LFE
        }
        
        channel = position_to_channel.get(position.lower(), 0)
        logger.info(f"Playing audio to position '{position}' mapped to channel {channel}")
        self.play_on_channel(audio_data, sample_rate, channel, effects_config)
    
    async def play_to_channel(self, audio_data: np.ndarray, channel: int, sample_rate: int, effects_config: Optional[Dict[str, Any]] = None):
        """Async wrapper for play_on_channel."""
        def _play():
            self.play_on_channel(audio_data, sample_rate, channel, effects_config)
        
        await asyncio.get_event_loop().run_in_executor(None, _play)
    
    async def play_to_position(self, audio_data: np.ndarray, position: str, sample_rate: int, effects_config: Optional[Dict[str, Any]] = None):
        """Async wrapper for play_to_position with optional effects."""
        def _play():
            self.play_to_position_sync(audio_data, position, sample_rate, effects_config)
        
        await asyncio.get_event_loop().run_in_executor(None, _play)
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.py_audio.terminate()
        logger.info("Audio player resources cleaned up")