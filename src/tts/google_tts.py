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
        voice_name: str = "en-US-Casual-K",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        use_ssml: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text using Google Cloud TTS with Chirp 3 HD voices."""
        
        # Check if this is a Chirp 3 HD voice (they don't support SSML or pitch)
        is_chirp3_hd = 'Chirp3-HD' in voice_name
        
        # Create input text object - Chirp 3 HD voices only support plain text
        if is_chirp3_hd:
            input_text = texttospeech.SynthesisInput(text=text)
        else:
            # Use SSML for legacy voices if requested
            if use_ssml:
                input_text = texttospeech.SynthesisInput(ssml=text)
            else:
                input_text = texttospeech.SynthesisInput(text=text)
        
        # Configure voice - Chirp 3 HD voices use different language code format
        if voice_name.startswith(("en-US-Chirp3-HD", "en-US-Casual")):
            language_code = "en-US"
        elif voice_name.startswith("en-GB-Chirp3-HD"):
            language_code = "en-GB"
        else:
            # Fallback to extracting from voice name
            language_code = voice_name[:5]
            
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        # Configure audio - Chirp 3 HD voices don't support pitch parameters
        if is_chirp3_hd:
            # Chirp 3 HD voices - no pitch support
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=speaking_rate,
                sample_rate_hertz=self.sample_rate,
            )
        else:
            # Legacy voices - include pitch
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
            f"Synthesized {len(audio_bytes)} bytes of audio using voice '{voice_name}' "
            f"for text of length {len(text)} as {audio_segment.channels}-channel audio"
        )
        
        return audio_array, audio_segment.frame_rate