"""RAG-enhanced multi-agent manager for knowledge-grounded conversations."""

import asyncio
import re
from typing import Dict, List, Optional

from src.knowledge.rag import KnowledgeBase
from src.llm.gemini import GeminiClient
from src.rag_enhanced_agent import RagEnhancedAgent
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.multi_channel_player import MultiChannelPlayer

# Set up module logger
logger = setup_logger("rag_manager")

class RagMultiAgentManager:
    """Manager for multiple RAG-enhanced speaking agents."""
    
    def __init__(self):
        """Initialize the RAG-enhanced multi-agent manager."""
        # Load configurations
        self.agents_config = load_config("agents")
        
        # Initialize shared components
        self.llm_client = GeminiClient()
        self.tts_client = GoogleTTSClient()
        self.knowledge_base = KnowledgeBase()
        self.audio_player = MultiChannelPlayer()
        
        # Create agents
        self.agents: Dict[str, RagEnhancedAgent] = {}
        
        # Conversation state
        self.context = []
        self.speaking = False
        
        logger.info("RAG multi-agent manager initialized")
    
    async def start(self) -> None:
        """Start the manager and initialize components."""
        # Start audio player
        await self.audio_player.start()
        
        # Initialize agents
        for agent_id, config in self.agents_config.items():
            self.agents[agent_id] = RagEnhancedAgent(
                agent_id=agent_id,
                config=config,
                knowledge_base=self.knowledge_base,
                llm_client=self.llm_client,
                tts_client=self.tts_client
            )
        
        logger.info(f"RAG multi-agent manager started with {len(self.agents)} agents")
    
    async def stop(self) -> None:
        """Stop the manager and clean up resources."""
        self.audio_player.cleanup()
        logger.info("RAG multi-agent manager stopped")
    
    async def ingest_knowledge(self) -> None:
        """Ingest knowledge from the document directory."""
        try:
            print("Ingesting knowledge documents...")
            count = await self.knowledge_base.ingest_directory()
            print(f"Successfully ingested {count} documents into the knowledge base.")
            
        except Exception as e:
            logger.error(f"Error ingesting knowledge: {str(e)}")
            print(f"Error: {str(e)}")
    
    async def run_conversation(self, topic: str, turns: int = 3) -> None:
        """Run a knowledge-enhanced conversation between agents."""
        logger.info(f"Starting RAG-enhanced conversation on topic: {topic}")
        
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
    
    async def _speak(self, agent: RagEnhancedAgent, ssml_text: str) -> None:
        """Have an agent speak the given text."""
        try:
            self.speaking = True
            
            # Debug logging for agent identity
            logger.info(f"_speak called for agent: {agent.name} (ID: {agent.agent_id})")
            logger.info(f"Agent {agent.name} - position: {getattr(agent, 'position', 'None')}, audio_channel: {getattr(agent, 'audio_channel', 'None')}")
            
            # Check if this agent uses a Chirp 3 HD voice (they don't support SSML)
            is_chirp3_hd = 'Chirp3-HD' in agent.voice_name
            
            # Convert to speech
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=ssml_text,
                voice_name=agent.voice_name,
                speaking_rate=agent.speaking_rate,
                pitch=agent.pitch,
                use_ssml=not is_chirp3_hd  # Use SSML only for non-Chirp3-HD voices
            )
            
            # Play to the appropriate channel with audio effects if configured
            effects_config = getattr(agent, 'audio_effects', {})
            
            if hasattr(agent, 'position') and agent.position:
                logger.info(f"Playing {agent.name} to position: {agent.position}")
                if effects_config:
                    logger.info(f"Applying audio effects to {agent.name}: {list(effects_config.keys())}")
                await self.audio_player.play_to_position(
                    audio_data, agent.position, sample_rate, effects_config
                )
            else:
                logger.info(f"Playing {agent.name} to channel: {agent.audio_channel}")
                if effects_config:
                    logger.info(f"Applying audio effects to {agent.name}: {list(effects_config.keys())}")
                await self.audio_player.play_to_channel(
                    audio_data, agent.audio_channel, sample_rate, effects_config
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