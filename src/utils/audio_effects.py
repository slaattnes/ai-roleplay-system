"""Audio effects processing for voice enhancement."""

import numpy as np
from typing import Optional, Dict, Any
from src.utils.logging import setup_logger

logger = setup_logger("audio_effects")

class AudioEffects:
    """Audio effects processor for voice enhancement."""
    
    def __init__(self):
        """Initialize audio effects processor."""
        pass
    
    def apply_echo(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        delay_ms: float = 300.0,
        decay: float = 0.4,
        mix: float = 0.3
    ) -> np.ndarray:
        """
        Apply echo effect to audio data.
        
        Args:
            audio_data: Input audio as numpy array
            sample_rate: Sample rate in Hz
            delay_ms: Echo delay in milliseconds
            decay: Echo decay factor (0.0-1.0)
            mix: Mix level of echo with original (0.0-1.0)
            
        Returns:
            Audio data with echo effect applied
        """
        try:
            # Ensure input is proper dtype
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Convert delay from milliseconds to samples
            delay_samples = int((delay_ms / 1000.0) * sample_rate)
            
            # Ensure we have enough data for the delay
            if delay_samples >= len(audio_data):
                logger.warning(f"Echo delay ({delay_ms}ms) too long for audio length, reducing")
                delay_samples = max(1, len(audio_data) // 4)  # Use quarter of the audio length
            
            # Work with float32 for processing to avoid overflow
            audio_float = audio_data.astype(np.float32)
            
            # Create output buffer - don't extend, just overlay echo
            output = audio_float.copy()
            
            # Apply echo by overlaying delayed signal
            for i in range(delay_samples, len(audio_float)):
                echo_sample = audio_float[i - delay_samples] * decay * mix
                output[i] = output[i] + echo_sample
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output))
            if max_val > 32000:  # Leave headroom for int16
                output = output * (32000 / max_val)
            
            logger.debug(f"Applied echo: delay={delay_ms}ms, decay={decay}, mix={mix}")
            
            return output.astype(np.int16)
            
        except Exception as e:
            logger.error(f"Error applying echo effect: {e}")
            return audio_data
    
    def apply_reverb(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        room_size: float = 0.5,
        damping: float = 0.5,
        mix: float = 0.3
    ) -> np.ndarray:
        """
        Apply simple reverb effect to audio data.
        
        Args:
            audio_data: Input audio as numpy array
            sample_rate: Sample rate in Hz
            room_size: Room size simulation (0.0-1.0)
            damping: High frequency damping (0.0-1.0)
            mix: Mix level of reverb with original (0.0-1.0)
            
        Returns:
            Audio data with reverb effect applied
        """
        try:
            # Simple multi-tap delay reverb
            delays_ms = [50, 100, 150, 200, 250] * room_size
            decays = [0.6, 0.5, 0.4, 0.3, 0.2]
            
            output = audio_data.copy().astype(np.float32)
            
            for delay_ms, decay in zip(delays_ms, decays):
                delay_samples = int((delay_ms / 1000.0) * sample_rate)
                if delay_samples < len(audio_data):
                    # Create delayed signal
                    delayed = np.zeros_like(output)
                    delayed[delay_samples:] = audio_data[:-delay_samples] if delay_samples > 0 else audio_data
                    
                    # Apply damping (simple low-pass filter)
                    if damping > 0:
                        for i in range(1, len(delayed)):
                            delayed[i] = delayed[i] * (1 - damping) + delayed[i-1] * damping
                    
                    # Mix in the delayed signal
                    output += delayed * decay * mix
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 0.95:
                output = output * (0.95 / max_val)
            
            logger.debug(f"Applied reverb: room_size={room_size}, damping={damping}, mix={mix}")
            
            return output.astype(audio_data.dtype)
            
        except Exception as e:
            logger.error(f"Error applying reverb effect: {e}")
            return audio_data
    
    def apply_effects(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        effects_config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply multiple effects based on configuration.
        
        Args:
            audio_data: Input audio as numpy array
            sample_rate: Sample rate in Hz
            effects_config: Dictionary of effects and their parameters
            
        Returns:
            Audio data with effects applied
        """
        processed_audio = audio_data.copy()
        
        # Apply echo if configured
        if "echo" in effects_config:
            echo_params = effects_config["echo"]
            processed_audio = self.apply_echo(
                processed_audio,
                sample_rate,
                delay_ms=echo_params.get("delay_ms", 300.0),
                decay=echo_params.get("decay", 0.4),
                mix=echo_params.get("mix", 0.3)
            )
        
        # Apply reverb if configured
        if "reverb" in effects_config:
            reverb_params = effects_config["reverb"]
            processed_audio = self.apply_reverb(
                processed_audio,
                sample_rate,
                room_size=reverb_params.get("room_size", 0.5),
                damping=reverb_params.get("damping", 0.5),
                mix=reverb_params.get("mix", 0.3)
            )
        
        return processed_audio
