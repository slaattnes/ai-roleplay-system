"""RAG-enhanced agent implementation."""

import asyncio
import re
from typing import Dict, List, Optional

from src.knowledge.rag import KnowledgeBase
from src.llm.gemini import GeminiClient
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("rag_enhanced_agent")

class RagEnhancedAgent:
    """Agent implementation with RAG capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        config: Dict,
        knowledge_base: KnowledgeBase,
        llm_client: GeminiClient,
        tts_client: GoogleTTSClient
    ):
        """Initialize a RAG-enhanced agent with its configuration."""
        self.agent_id = agent_id
        self.name = config["name"]
        self.role = config["role"]
        self.description = config["description"]
        self.voice_name = config["voice_name"]
        self.speaking_rate = config["voice_params"]["speaking_rate"]
        self.pitch = config["voice_params"].get("pitch", 0.0)  # Default to 0.0 if not specified
        
        # Load audio effects configuration if present
        self.audio_effects = config.get("audio_effects", {})
        self.position = config["position"]
        self.audio_channel = config["audio_channel"]
        
        # Knowledge and LLM components
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # Conversation context
        self.context = []
        
        logger.info(f"Initialized RAG-enhanced agent: {self.name} ({self.role})")
    
    async def generate_response(
        self, 
        topic: str, 
        context: List[str] = None
    ) -> str:
        """Generate a knowledge-enhanced response on the given topic."""
        system_instruction = self._build_system_instruction()
        
        # First, retrieve relevant knowledge
        knowledge = await self._retrieve_knowledge(topic, context)
        
        # Build prompt with knowledge
        prompt = self._build_prompt(topic, knowledge, context)
        
        # Generate response
        logger.info(f"Agent {self.name} generating response on: {topic}")
        response = await self.llm_client.generate_response(
            prompt=prompt,
            system_instruction=system_instruction
        )
        
        # Check if this agent uses a Chirp 3 HD voice (they don't support SSML)
        is_chirp3_hd = 'Chirp3-HD' in self.voice_name
        
        if is_chirp3_hd:
            # Return plain text for Chirp 3 HD voices
            return response.strip()
        else:
            # Process response to ensure proper SSML for legacy voices
            return self._process_response_for_ssml(response)
    
    async def _retrieve_knowledge(
        self, 
        topic: str, 
        context: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve relevant knowledge for the topic and context.
        
        Args:
            topic: Current conversation topic
            context: Recent conversation context
            
        Returns:
            Formatted string with retrieved knowledge
        """
        # Skip knowledge retrieval for now to speed up responses
        return "No additional knowledge needed for this discussion."
    
    def _build_system_instruction(self) -> str:
        """Build system instruction for the LLM."""
        return f"""
        You are {self.name}, a {self.role}. {self.description}
        
        Keep responses brief (2-3 sentences max) and in character.
        No need for SSML formatting - speak naturally.
        """
    
    def _build_prompt(
        self, 
        topic: str, 
        knowledge: str,
        context: Optional[List[str]] = None
    ) -> str:
        """
        Build a prompt with knowledge and context.
        
        Args:
            topic: Current conversation topic
            knowledge: Retrieved knowledge text
            context: Recent conversation context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Topic: {topic}\n\n"
        
        # Add conversation context if provided (keep minimal)
        if context and len(context) > 0:
            recent_context = context[-2:]  # Only last 2 exchanges
            context_text = "\n".join(recent_context)
            prompt += f"Recent conversation:\n{context_text}\n\n"
        
        prompt += "Give a brief response (2-3 sentences) from your perspective."
        
        return prompt
    
    def _process_response_for_ssml(self, response: str) -> str:
        """Process LLM response to ensure proper SSML formatting."""
        # Simplified SSML - just wrap in basic speak tags
        # Remove any existing SSML first
        import re
        text = re.sub(r'<[^>]+>', '', response).strip()
        
        # Escape XML characters
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Add simple SSML structure
        return f"<speak>{text}</speak>"