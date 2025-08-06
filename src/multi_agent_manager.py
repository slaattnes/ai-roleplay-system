"""Manager for multiple speaking agents with basic turn-taking."""

import asyncio
import re
from typing import Dict, List, Optional

from src.llm.gemini import GeminiClient
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.multi_channel_player import MultiChannelPlayer

# Set up module logger
logger = setup_logger("multi_agent_manager")

class Agent:
    """Individual agent in the multi-agent system."""
    
    def __init__(
        self,
        agent_id: str,
        config: Dict,
        llm_client: GeminiClient,
        tts_client: GoogleTTSClient
    ):
        """Initialize an agent with its configuration."""
        self.agent_id = agent_id
        self.name = config["name"]
        self.role = config["role"]
        self.description = config["description"]
        self.voice_name = config["voice_name"]
        self.speaking_rate = config["voice_params"]["speaking_rate"]
        self.pitch = config["voice_params"]["pitch"]
        self.position = config["position"]
        self.audio_channel = config["audio_channel"]
        
        # Shared clients
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # Conversation context
        self.context = []
        
        logger.info(f"Initialized agent: {self.name} ({self.role})")
    
    async def generate_response(self, topic: str, context: List[str] = None) -> str:
        """Generate a response on the given topic with optional context."""
        system_instruction = self._build_system_instruction()
        
        # Build prompt
        prompt = f"As {self.name}, a {self.role}, please respond to the topic: {topic}\n\n"
        
        # Add conversation context if provided
        if context and len(context) > 0:
            context_text = "\n".join(context)
            prompt += f"Recent conversation:\n{context_text}\n\n"
        
        # Generate response
        logger.info(f"Agent {self.name} generating response on: {topic}")
        response = await self.llm_client.generate_response(
            prompt=prompt,
            system_instruction=system_instruction
        )
        
        # Process response to ensure proper SSML
        return self._process_response_for_ssml(response)
    
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
        
        Keep your response concise, around 3-4 sentences.
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


class MultiAgentManager:
    """Manager for multiple speaking agents with turn-taking."""
    
    def __init__(self):
        """Initialize the multi-agent manager."""
        # Load configurations
        self.agents_config = load_config("agents")
        
        # Initialize shared components
        self.llm_client = GeminiClient()
        self.tts_client = GoogleTTSClient()
        self.audio_player = MultiChannelPlayer()
        
        # Create agents
        self.agents: Dict[str, Agent] = {}
        for agent_id, config in self.agents_config.items():
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                config=config,
                llm_client=self.llm_client,
                tts_client=self.tts_client
            )
        
        # Conversation state
        self.context = []
        self.speaking = False
        
        logger.info(f"Multi-agent manager initialized with {len(self.agents)} agents")
    
    async def start(self):
        """Start the multi-agent manager."""
        await self.audio_player.start()  # Add await here
        logger.info("Multi-agent manager started")
    
    async def stop(self) -> None:
        """Stop the multi-agent manager."""
        self.audio_player.cleanup()
        logger.info("Multi-agent manager stopped")
    
    async def run_conversation(self, topic: str, turns: int = 3) -> None:
        """Run a conversation between agents for a specified number of turns."""
        logger.info(f"Starting conversation on topic: {topic}")
        
        # Reset conversation context
        self.context = []
        
        try:
            # Get the agent IDs as a list for turn-taking
            agent_ids = list(self.agents.keys())
            
            for turn in range(turns):
                for agent_id in agent_ids:
                    agent = self.agents[agent_id]
                    
                    # Generate response
                    ssml_response = await agent.generate_response(
                        topic=topic,
                        context=self.context[-5:] if self.context else None
                    )
                    
                    # Extract plain text for context
                    plain_text = self._extract_plain_text(ssml_response)
                    self.context.append(f"{agent.name}: {plain_text}")
                    
                    # Speak the response
                    print(f"\n{agent.name} is speaking...")
                    await self._speak(agent, ssml_response)
                    
                    # Small pause between agents
                    await asyncio.sleep(1)
            
            logger.info("Conversation complete")
            
        except Exception as e:
            logger.error(f"Error in conversation: {str(e)}")
    
    async def _speak(self, agent: Agent, ssml_text: str) -> None:
        """Have an agent speak the given text."""
        try:
            self.speaking = True
            
            # Convert to speech
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=ssml_text,
                voice_name=agent.voice_name,
                speaking_rate=agent.speaking_rate,
                pitch=agent.pitch,
                use_ssml=True
            )
            
            # Play to the appropriate channel
            if hasattr(agent, 'position') and agent.position:
                self.audio_player.play_to_position(
                    audio_data, agent.position, sample_rate
                )
            else:
                self.audio_player.play_on_channel(
                    audio_data, sample_rate, agent.audio_channel
                )
            
            self.speaking = False
            
        except Exception as e:
            logger.error(f"Error in _speak: {str(e)}")
            self.speaking = False
    
    def _extract_plain_text(self, ssml: str) -> str:
        """Extract plain text from SSML."""
        # Remove all SSML tags
        text = re.sub(r'<[^>]+>', '', ssml)
        
        # Fix common SSML entities
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        
        return text.strip()