"""Single agent implementation that can speak."""

import asyncio
import re
from typing import Optional

from src.llm.gemini import GeminiClient
from src.tts.google_tts import GoogleTTSClient
from src.utils.audio_player import AudioPlayer
from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("single_agent")

class SingleSpeakingAgent:
    """A simple agent that can generate and speak text."""
    
    def __init__(self, agent_id: str = "agent1"):
        """Initialize the agent with a specific configuration."""
        # Load agent configuration
        self.agents_config = load_config("agents")
        self.agent_config = self.agents_config[agent_id]
        
        # Extract agent details
        self.name = self.agent_config["name"]
        self.role = self.agent_config["role"]
        self.description = self.agent_config["description"]
        self.voice_name = self.agent_config["voice_name"]
        self.speaking_rate = self.agent_config["voice_params"]["speaking_rate"]
        self.pitch = self.agent_config["voice_params"]["pitch"]
        self.output_channel = self.agent_config["voice_params"].get("output_channel") # Get output channel override
        
        # Initialize components
        self.llm_client = GeminiClient()
        self.tts_client = GoogleTTSClient()
        self.audio_player = AudioPlayer()
        
        logger.info(f"Initialized agent: {self.name} ({self.role})")
    
    async def speak_on_topic(self, topic: str) -> None:
        """Generate and speak a response on a given topic."""
        try:
            # Generate response using LLM
            system_instruction = self._build_system_instruction()
            prompt = f"As {self.name}, a {self.role}, please share your thoughts on the topic: {topic}"
            
            logger.info(f"Generating response for topic: {topic}")
            response = await self.llm_client.generate_response(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            # Process response to add SSML tags if not present
            ssml_response = self._process_response_for_ssml(response)
            
            # Convert to speech
            logger.info("Converting text to speech")
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=ssml_response,
                voice_name=self.voice_name,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
                use_ssml=True
            )
            
            # Play the audio
            logger.info("Playing audio")
            await self.audio_player.play_audio(audio_data, sample_rate, output_channel=self.output_channel)
            
            logger.info("Response complete")
            
        except Exception as e:
            logger.error(f"Error in speak_on_topic: {str(e)}")
    
    def _build_system_instruction(self) -> str:
        """Build system instruction for the LLM."""
        return f"""
        You are {self.name}, a {self.role}. 
        {self.description}
        
        Your responses should be in character and reflect your expertise and persona.
        
        IMPORTANT: Format your response with SSML tags to make your speech more natural.
        Use <break>, <prosody>, and other SSML tags to add pauses, emphasis, and intonation.
        
        EXAMPLE SSML FORMATTING:
        <speak>
          <prosody rate="medium" pitch="+0st">
            This is an important point. <break time="300ms"/> Let me elaborate.
          </prosody>
          <prosody rate="medium" pitch="+2st">
            I'm particularly interested in this aspect!
          </prosody>
        </speak>
        """
    
    def _process_response_for_ssml(self, response: str) -> str:
        """Process LLM response to ensure proper SSML formatting."""
        # Check if response already has <speak> tags
        if "<speak>" in response and "</speak>" in response:
            # Extract content between speak tags
            match = re.search(r"<speak>(.*?)</speak>", response, re.DOTALL)
            if match:
                ssml_content = match.group(1).strip()
                return f"<speak>{ssml_content}</speak>"
        
        # If no speak tags, wrap the whole response
        # But first, escape any XML/HTML that might be in the text
        text = response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Add basic SSML structure
        return f"""
        <speak>
            <prosody rate="{self.speaking_rate}" pitch="{self.pitch:+.1f}st">
                {text}
            </prosody>
        </speak>
        """
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.audio_player.cleanup()