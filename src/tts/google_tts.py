"""Google Cloud Text-to-Speech API integration."""

import asyncio
import io
import re
from typing import Dict, Optional, Tuple

import numpy as np
from google.cloud import texttospeech
from pydub import AudioSegment

from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("google_tts")

class GoogleTTSClient:
    """Client for Google Cloud Text-to-Speech API."""
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize the TTS client."""
        self.client = texttospeech.TextToSpeechClient()
        self.sample_rate = sample_rate
        logger.info(f"Initialized Google TTS client with sample rate {sample_rate}Hz")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_name: str = "en-US-Neural2-F",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        sample_rate_hertz: int = 24000,
        use_ssml: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Google Cloud TTS.
        
        Args:
            text: Text or SSML to synthesize
            voice_name: Google TTS voice name
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Voice pitch (-20.0 to 20.0)
            sample_rate_hertz: Audio sample rate
            use_ssml: Whether the input text is SSML
            
        Returns:
            Tuple of (audio data as numpy array, sample rate)
        """
        # Fix common SSML issues - add this to help with Chirp voices
        if use_ssml:
            # Clean up the SSML to ensure it's valid
            text = self._clean_ssml(text)
            logger.debug(f"Processed SSML: {text[:100]}...")
        
        # Extract language code from voice name
        if "-" in voice_name:
            language_code = "-".join(voice_name.split("-")[:2])  # e.g., "en-US" from "en-US-Neural2-F"
        else:
            # Default to English if can't extract
            language_code = "en-US"
            logger.warning(f"Could not extract language code from {voice_name}, using {language_code}")
        
        # Create input text object based on whether we're using SSML or plain text
        if use_ssml:
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        # Configure audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
            sample_rate_hertz=sample_rate_hertz,
            effects_profile_id=["medium-bluetooth-speaker-class-device"]
        )
        
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.synthesize_speech(
                    input=input_text, voice=voice, audio_config=audio_config
                )
            )
            
            # Process audio data
            audio_bytes = response.audio_content
            
            # Convert to numpy array
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            audio_array = np.array(audio_segment.get_array_of_samples())
            
            logger.debug(
                f"Synthesized {len(audio_bytes)} bytes of audio "
                f"for text of length {len(text)}"
            )
            
            return audio_array, audio_segment.frame_rate
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"TTS synthesis error: {error_message}")
            
            # If it's an SSML-related error, raise it so the caller can handle it
            if "does not support SSML" in error_message:
                raise e
            
            # For other errors, return a short silence instead of crashing
            silence = np.zeros(int(sample_rate_hertz * 0.5), dtype=np.int16)  # 0.5 seconds of silence
            logger.warning("Returning silence due to TTS error")
            return silence, sample_rate_hertz
    
    def _clean_ssml(self, ssml: str) -> str:
        """
        Clean and validate SSML to ensure it works with Google TTS.
        
        Args:
            ssml: Raw SSML text
            
        Returns:
            Cleaned SSML text
        """
        # Ensure it has proper speak tags
        if "<speak>" not in ssml or "</speak>" not in ssml:
            # Wrap in speak tags if missing
            ssml = f"<speak>{ssml}</speak>"
        
        # Extract content between speak tags to clean it
        speak_match = re.search(r"<speak>(.*?)</speak>", ssml, re.DOTALL)
        if not speak_match:
            # Shouldn't happen but just in case
            return f"<speak>{ssml}</speak>"
        
        content = speak_match.group(1)
        
        # Remove any unsupported or malformed SSML tags
        # For Chirp voices, simplify to basic tags
        
        # 1. Fix common issues with prosody tags
        content = re.sub(r'<prosody\s+rate="([^"]+)"\s+pitch="([^"]+)">', 
                       r'<prosody rate="\1">', content)
        
        # 2. Remove pitch attributes as they might not be supported in Chirp
        content = re.sub(r'pitch="[^"]+"', '', content)
        
        # 3. Fix common malformed tags
        content = re.sub(r'<([^>]+)>\s*<\/\1>\s*', '', content)  # Remove empty tags
        
        # 4. Ensure break tags are well-formed
        content = re.sub(r'<break\s+([^>]*)(?<!/)>', r'<break \1/>', content)
        
        # Return cleaned SSML
        return f"<speak>{content}</speak>"