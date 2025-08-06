"""Basic audio playback utility."""

import asyncio
from typing import Optional

import numpy as np
import pyaudio

from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("audio_player")

class AudioPlayer:
    """Simple audio playback handler."""
    
    def __init__(self):
        """Initialize the audio player."""
        # Load main configuration
        self.main_config = load_config("main")
        
        # Audio device settings
        device_setting = self.main_config["audio"]["device_index"]
        if isinstance(device_setting, str):
            if device_setting.lower() == "default":
                self.device_index = None
            else:
                try:
                    self.device_index = int(device_setting)
                except ValueError:
                    logger.error(f"Invalid audio device_index: {device_setting}. Must be an integer or 'default'.")
                    self.device_index = None # Fallback to default
        else:
            self.device_index = device_setting # Should be an int or None

        self.sample_rate = self.main_config["audio"]["sample_rate"]
        self.channels = self.main_config["audio"]["channels"]
        
        # PyAudio instance
        self.py_audio = pyaudio.PyAudio()
        
        logger.info(f"Initialized audio player with device index {self.device_index if self.device_index is not None else 'default'}")
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int, output_channel: Optional[int] = None) -> None:
        """Play audio data through the configured output device."""
        try:
            if output_channel is not None:
                # Create multi-channel audio with sound only on the specified channel
                total_channels = self.channels
                samples_per_channel = len(audio_data)
                
                # Create a silent multi-channel array
                multi_channel_data = np.zeros((samples_per_channel * total_channels,), dtype=np.int16)
                
                # Place the mono audio data into the specified channel
                # Audio data is interleaved, so we need to place samples at the right indices
                for i in range(samples_per_channel):
                    multi_channel_data[i * total_channels + output_channel] = audio_data[i]
                
                playback_channels = total_channels
                playback_data = multi_channel_data
                logger.info(f"Playing mono audio on channel {output_channel} of {playback_channels} channels at {sample_rate}Hz.")
            else:
                # Use the audio data as-is
                playback_channels = self.channels
                playback_data = audio_data
                logger.info(f"Playing audio at {sample_rate}Hz with {playback_channels} channels.")
            
            # Open stream
            try:
                self.stream = self.py_audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.device_index
                )
            except OSError as e:
                logger.error(f"Error starting audio stream: {e}")
                logger.info("Falling back to stereo (2 channels)")
                self.channels = 2  # Fallback to stereo
                self.stream = self.py_audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.device_index
                )
            
            # Write audio data
            self.stream.write(playback_data.tobytes())
            
            # Close stream
            self.stream.stop_stream()
            self.stream.close()
            
            logger.debug(f"Played {len(audio_data)} audio samples")
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up PyAudio resources."""
        self.py_audio.terminate()
        logger.info("Audio player resources cleaned up")