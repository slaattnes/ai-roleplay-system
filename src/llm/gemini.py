"""Google Gemini API integration for agent language generation."""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.utils.config import get_env_var
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("gemini")

# Initialize Gemini with API key
genai.configure(api_key=get_env_var("GOOGLE_API_KEY"))

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    
    def __init__(
        self, 
        model: str = "gemini-1.5-pro", 
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
    ):
        """Initialize the Gemini client."""
        self.model = model
        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        self.model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config
        )
        logger.info(f"Initialized Gemini client with model: {model}")
    
    async def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a response from the Gemini model."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Prepare the chat session
                chat = self.model_instance.start_chat(history=[])
                
                # Add system instruction if provided
                if system_instruction:
                    chat.send_message(f"[SYSTEM INSTRUCTION]\n{system_instruction}")
                
                # Send the actual prompt and get response
                response = chat.send_message(prompt)
                text_response = response.text
                
                return text_response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to generate response after {max_retries} attempts")
                    raise