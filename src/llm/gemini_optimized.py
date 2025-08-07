"""Optimized Google Gemini API integration with parallel generation support."""

import asyncio
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

class OptimizedGeminiClient:
    """Optimized client for Google Gemini API with parallel generation support."""
    
    def __init__(
        self, 
        model: str = "gemini-1.5-flash",  # Use flash model for faster responses
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_output_tokens: int = 512,  # Reduced for faster generation
    ):
        """Initialize the optimized Gemini client."""
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
        
        # Connection pool for parallel requests
        self.active_requests = 0
        self.max_concurrent_requests = 5
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        logger.info(f"Initialized optimized Gemini client with model: {model}")
    
    async def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a response from the Gemini model with optimizations."""
        async with self.request_semaphore:  # Limit concurrent requests
            return await self._generate_single_response(prompt, system_instruction)
    
    async def generate_multiple_responses(
        self,
        prompts_and_instructions: List[Tuple[str, Optional[str]]]
    ) -> List[str]:
        """Generate multiple responses in parallel."""
        tasks = [
            self.generate_response(prompt, instruction)
            for prompt, instruction in prompts_and_instructions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate response {i}: {str(result)}")
                responses.append("I apologize, but I'm having trouble generating a response right now.")
            else:
                responses.append(result)
        
        return responses
    
    async def _generate_single_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a single response with optimized retry logic."""
        max_retries = 2  # Reduced retries for speed
        retry_delay = 1   # Faster retry
        
        for attempt in range(max_retries):
            try:
                self.active_requests += 1
                
                # Create a new model instance for this request to avoid session conflicts
                model_instance = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=self.generation_config
                )
                
                # Combine system instruction and prompt for efficiency
                full_prompt = prompt
                if system_instruction:
                    full_prompt = f"[SYSTEM INSTRUCTION]\n{system_instruction}\n\n{prompt}"
                
                # Run in executor to avoid blocking
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model_instance.generate_content(full_prompt)
                )
                
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Gentler backoff
                else:
                    logger.error(f"Failed to generate response after {max_retries} attempts")
                    raise
            finally:
                self.active_requests -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.max_concurrent_requests,
            "model": self.model
        }


# Backwards compatibility - alias the original class name
GeminiClient = OptimizedGeminiClient
