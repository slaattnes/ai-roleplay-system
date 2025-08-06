"""Google Cloud Text-to-Speech API integration."""

import asyncio
import io
from typing import Dict, Optional, Tuple

import numpy as np
from google.cloud import texttospeech
from pydub import AudioSegment

from src.utils.logging import setup_logger
from src.utils.config import load_config

# Set up module logger
logger = setup_logger("google_tts")

class GoogleTTSClient:
    """Client for Google Cloud Text-to-Speech API."""
    
    def __init__(self):
        """Initialize the TTS client."""
        self.main_config = load_config("main")
        self.sample_rate = self.main_config["audio"]["sample_rate"]
        self.client = texttospeech.TextToSpeechClient()
        logger.info(f"Initialized Google TTS client with sample rate {self.sample_rate}Hz")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_name: str = "en-US-Neural2-F",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        use_ssml: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text using Google Cloud TTS."""
        # Create input text object based on whether we're using SSML or plain text
        if use_ssml:
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_name[:5],  # Extract language code (e.g., "en-US")
            name=voice_name
        )
        
        # Configure audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
            sample_rate_hertz=self.sample_rate,
        )
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )
        )
        
        # Process audio data
        audio_bytes = response.audio_content
        
        # Convert to numpy array (keep as mono from Google TTS)
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        audio_array = np.array(audio_segment.get_array_of_samples())
        
        logger.debug(
            f"Synthesized {len(audio_bytes)} bytes of audio "
            f"for text of length {len(text)} as {audio_segment.channels}-channel audio"
        )
        
        return audio_array, audio_segment.frame_rate