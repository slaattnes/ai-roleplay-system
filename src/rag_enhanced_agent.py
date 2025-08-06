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
        self.pitch = config["voice_params"]["pitch"]
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
        
        # Process response to ensure proper SSML
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
        # Combine topic and context for a better query
        query = topic
        if context and len(context) > 0:
            # Extract the most recent context entries
            recent_context = context[-3:]
            query = f"{topic} {' '.join(recent_context)}"
        
        # Query the knowledge base
        results = await self.knowledge_base.query(query, n_results=3)
        
        if not results:
            return "No specific knowledge retrieved."
        
        # Format the results
        knowledge_text = "Retrieved knowledge:\n\n"
        for i, result in enumerate(results):
            source = result["metadata"].get("title", "Unknown source")
            knowledge_text += f"[Source {i+1}: {source}]\n{result['text']}\n\n"
        
        return knowledge_text
    
    def _build_system_instruction(self) -> str:
        """Build system instruction for the LLM."""
        return f"""
        You are {self.name}, a {self.role}. 
        {self.description}
        
        Your responses should be in character and reflect your expertise and persona.
        
        When responding, incorporate the provided knowledge naturally into your response.
        Do not mention "According to the retrieved knowledge" or similar phrases.
        Speak as if this knowledge is part of your own expertise.
        
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
        
        Keep your response concise, around 3-5 sentences.
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
        prompt = f"As {self.name}, please respond to the topic: {topic}\n\n"
        
        # Add knowledge
        prompt += f"{knowledge}\n\n"
        
        # Add conversation context if provided
        if context and len(context) > 0:
            context_text = "\n".join(context[-5:])  # Use most recent 5 entries
            prompt += f"Recent conversation:\n{context_text}\n\n"
        
        # Add specific instructions
        prompt += """
        In your response:
        1. Incorporate the retrieved knowledge naturally
        2. Stay in character as your persona
        3. React to what others have said in the conversation
        4. Express your unique perspective on the topic
        """
        
        return prompt
    
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