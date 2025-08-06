"""Multi-agent manager with orchestrator integration for advanced conversation logic."""

import asyncio
import re
from typing import Dict, List, Optional

from src.llm.gemini import GeminiClient
from src.orchestrator.orchestrator import Orchestrator
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.multi_channel_player import MultiChannelPlayer

# Set up module logger
logger = setup_logger("orchestrated_manager")

class OrchestratedAgent:
    """Individual agent with orchestrator integration."""
    
    def __init__(
        self,
        agent_id: str,
        config: Dict,
        orchestrator: Orchestrator,
        llm_client: GeminiClient,
        tts_client: GoogleTTSClient
    ):
        """Initialize an agent with orchestrator integration."""
        self.agent_id = agent_id
        self.name = config["name"]
        self.role = config["role"]
        self.description = config["description"]
        self.voice_name = config["voice_name"]
        
        # Extract voice parameters with defaults for Chirp 3 HD voices
        voice_params = config.get("voice_params", {})
        self.speaking_rate = voice_params.get("speaking_rate", 1.0)
        # Make pitch optional with a default of 0.0
        self.pitch = voice_params.get("pitch", 0.0)
        
        self.position = config["position"]
        self.audio_channel = config["audio_channel"]
        
        # Determine if this is the moderator
        agent_name = config.get("name", "").lower()
        agent_role = config.get("role", "").lower()
        self.is_moderator = (
            agent_id == "agent5" or
            agent_id == "moderator" or
            "organ" in agent_name or
            agent_role == "moderator"
        )
        
        # Shared components
        self.orchestrator = orchestrator
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # Agent state
        self.is_speaking = False
        self.has_speaking_token = False
        
        logger.info(f"Initialized orchestrated agent: {self.name} ({self.role})")
    
    async def request_to_speak(self, reason: str) -> None:
        """
        Request permission to speak from the orchestrator.
        
        Args:
            reason: Reason for wanting to speak
        """
        await self.orchestrator.request_to_speak(self.agent_id, reason)
    
    async def release_speaking_token(self) -> None:
        """Release the speaking token back to the orchestrator."""
        if self.has_speaking_token:
            await self.orchestrator.release_speaking_token(self.agent_id)
            self.has_speaking_token = False
    
    async def speak(self, context: Dict) -> None:
        """
        Generate and speak a response based on the given context.
        
        Args:
            context: Speaking context
        """
        if not self.has_speaking_token:
            logger.warning(f"Agent {self.name} attempted to speak without token")
            return
        
        try:
            self.is_speaking = True
            
            # Generate response
            ssml_response = await self.generate_response(context)
            
            # Extract plain text for transcript
            plain_text = self._extract_plain_text(ssml_response)
            
            # Add to transcript
            await self.orchestrator.add_transcript(
                self.agent_id, self.name, plain_text
            )
            
            # Synthesize speech
            # Check if this is a Chirp HD voice that doesn't support SSML
            if "Chirp3-HD" in self.voice_name:
                logger.info(f"Chirp HD voice detected ({self.voice_name}), using plain text directly")
                # Extract plain text from SSML and use it directly
                plain_text = self._extract_plain_text(ssml_response)
                audio_data, sample_rate = await self.tts_client.synthesize_speech(
                    text=plain_text,
                    voice_name=self.voice_name,
                    speaking_rate=self.speaking_rate,
                    pitch=self.pitch,
                    sample_rate_hertz=48000,
                    use_ssml=False
                )
            else:
                # Try SSML first for non-Chirp voices
                try:
                    audio_data, sample_rate = await self.tts_client.synthesize_speech(
                        text=ssml_response,
                        voice_name=self.voice_name,
                        speaking_rate=self.speaking_rate,
                        pitch=self.pitch,
                        sample_rate_hertz=48000,
                        use_ssml=True
                    )
                except Exception as e:
                    if "does not support SSML" in str(e):
                        logger.info(f"SSML not supported for {self.voice_name}, falling back to plain text")
                        logger.info(f"Using speaking rate: {self.speaking_rate} for fallback")
                        # Fallback to plain text by extracting text from SSML
                        plain_text_fallback = self._extract_plain_text(ssml_response)
                        audio_data, sample_rate = await self.tts_client.synthesize_speech(
                            text=plain_text_fallback,
                            voice_name=self.voice_name,
                            speaking_rate=self.speaking_rate,
                            pitch=self.pitch,
                            sample_rate_hertz=48000,
                            use_ssml=False
                        )
                    else:
                        raise e
            
            # Return the audio data for playback by the manager
            self.is_speaking = False
            return audio_data, sample_rate, self.position
            
        except Exception as e:
            logger.error(f"Error in speak for {self.name}: {str(e)}")
            self.is_speaking = False
            return None, None, None
    
    async def generate_response(self, context: Dict) -> str:
        """
        Generate a response based on the context.
        
        Args:
            context: Speaking context with event type, topic, etc.
            
        Returns:
            SSML-formatted response
        """
        # Get current event type, topic, and description
        event_type = context.get("event_type", "")
        topic = context.get("topic", "")
        description = context.get("description", "")
        
        # Build system instruction
        system_instruction = self._build_system_instruction(event_type)
        
        # Get recent conversation history
        recent_transcripts = self.orchestrator.get_recent_transcripts()
        history_text = "\n".join([
            f"{t['speaker']}: {t['text']}" 
            for t in recent_transcripts
        ])
        
        # Build prompt
        prompt = self._build_prompt(event_type, topic, history_text)
        
        # Generate response
        logger.info(f"Agent {self.name} generating response for {event_type} on: {topic}")
        response = await self.llm_client.generate_response(
            prompt=prompt,
            system_instruction=system_instruction
        )
        
        # Process response to ensure proper SSML
        return self._process_response_for_ssml(response)
    
    def _build_system_instruction(self, event_type: str) -> str:
        """Build system instruction based on event type and role."""
        base_instruction = f"""
        You are {self.name}, a {self.role}. 
        {self.description}
        
        Your responses should be in character and reflect your expertise and persona.
        
        IMPORTANT: Format your response with ONLY SIMPLE SSML tags to make your speech more natural.
        Use ONLY <break time="300ms"/> for pauses and <emphasis> for emphasis.
        DO NOT use prosody, pitch, or other complex tags as they aren't supported in this voice.
        
        EXAMPLE SSML FORMATTING:
        <speak>
          Welcome to the discussion. <break time="300ms"/> Today we'll explore an important topic.
          <emphasis>This is a key point</emphasis> that I want to highlight.
        </speak>
        """
        
        # Add moderator-specific instructions
        if self.is_moderator:
            base_instruction += """
            
            As the moderator, you should:
            1. Guide the conversation and ensure all participants have a chance to speak
            2. Acknowledge and invite others when they indicate they want to speak
            3. Keep the discussion on topic and intervene politely if needed
            4. Summarize key points and facilitate transitions between topics
            """
        
        # Add event-specific instructions
        if event_type == "welcome" or event_type == "opening_remarks":
            base_instruction += """
            This is the opening of the session. Introduce the topic and set the tone.
            As the moderator, welcome everyone to the discussion and briefly introduce the topic.
            """
        elif event_type == "closing_remarks":
            base_instruction += """
            This is the closing of the session. Summarize key points discussed and offer
            concluding thoughts. Thank the participants.
            """
        elif event_type == "recess":
            base_instruction += """
            This is a brief recess. Engage in lighter, more informal conversation.
            You can briefly reflect on the discussion so far or mention related topics.
            """
        
        return base_instruction
    
    def _build_prompt(self, event_type: str, topic: str, history: str) -> str:
        """Build prompt based on event type and context."""
        prompt = f"As {self.name}, please respond to the topic: {topic}\n\n"
        
        # Add conversation history
        if history:
            prompt += f"Recent conversation:\n{history}\n\n"
        
        # Add event-specific instructions
        if event_type == "structured_discussion":
            prompt += """
            This is a structured discussion. Provide your expert perspective on the topic.
            Be clear, focused, and substantive.
            """
        elif event_type == "open_forum":
            prompt += """
            This is an open forum. You can respond more freely, ask questions, or
            challenge points made by others. Show your personality and opinions.
            """
        elif event_type == "presentation":
            prompt += """
            You are giving a brief presentation. Present your key insights on the topic
            in a clear and engaging manner. Structure your thoughts logically.
            """
        elif event_type == "panel_discussion":
            prompt += """
            This is a panel discussion. Respond to the points raised by others and
            add your own perspective. Be collaborative but don't hesitate to
            respectfully disagree when appropriate.
            """
        elif event_type == "keynote":
            prompt += """
            This is a keynote presentation. Deliver an insightful, thought-provoking
            perspective on the topic. Be eloquent and impactful.
            """
        
        # Add moderator-specific instructions for certain events
        if self.is_moderator:
            if event_type == "welcome" or event_type == "opening_remarks":
                prompt += """
                As the moderator, introduce the session and the topic. Briefly explain
                the format of the discussion and set a positive, intellectual tone.
                """
            elif event_type == "closing_remarks":
                prompt += """
                As the moderator, summarize the key points discussed today, acknowledge
                insights shared by others, and bring the session to a thoughtful close.
                """
        
        # Keep response concise
        prompt += "\n\nKeep your response concise, around 3-4 sentences."
        
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
        
        # Add basic SSML structure - very simple for maximum compatibility
        return f"<speak>{text}</speak>"
    
    def _extract_plain_text(self, ssml: str) -> str:
        """Extract plain text from SSML."""
        # Remove all SSML tags
        text = re.sub(r'<[^>]+>', '', ssml)
        
        # Fix common SSML entities
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        
        return text.strip()


class OrchestratedAgentManager:
    """Manager for agents with orchestrator integration."""
    
    def __init__(self):
        """Initialize the orchestrated agent manager."""
        # Load configurations
        self.agents_config = load_config("agents")
        
        # Initialize components
        self.orchestrator = Orchestrator()
        self.llm_client = GeminiClient()
        self.tts_client = GoogleTTSClient()
        self.audio_player = MultiChannelPlayer()
        
        # Agents will be created after orchestrator is started
        self.agents: Dict[str, OrchestratedAgent] = {}
        
        # Operation state
        self.is_running = False
        self.conversation_task = None
        
        logger.info("Orchestrated agent manager initialized")
    
    async def start(self) -> None:
        """Start the manager and its components."""
        try:
            # Start audio player - Fix: use await
            await self.audio_player.start()
            
            # Start orchestrator
            await self.orchestrator.start()
            
            # Create agents
            for agent_id, config in self.agents_config.items():
                self.agents[agent_id] = OrchestratedAgent(
                    agent_id=agent_id,
                    config=config,
                    orchestrator=self.orchestrator,
                    llm_client=self.llm_client,
                    tts_client=self.tts_client
                )
            
            # Start conversation management task
            self.is_running = True
            self.conversation_task = asyncio.create_task(self._manage_conversation())
            
            logger.info(f"Orchestrated agent manager started with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error starting orchestrated agent manager: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the manager and its components."""
        self.is_running = False
        
        # Cancel conversation task
        if self.conversation_task:
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Stop orchestrator
        await self.orchestrator.stop()
        
        # Stop audio player
        self.audio_player.cleanup()
        
        logger.info("Orchestrated agent manager stopped")
    
    async def _manage_conversation(self) -> None:
        """Main conversation management loop."""
        try:
            while self.is_running:
                # Check if we have a current speaker
                if not self.orchestrator.current_speaker:
                    # Get the next speaker from the orchestrator
                    next_speaker_id = await self.orchestrator.get_next_speaker()
                    
                    if next_speaker_id and next_speaker_id in self.agents:
                        # Grant speaking token to this agent
                        self.orchestrator.current_speaker = next_speaker_id
                        self.orchestrator.state.set_current_speaker(next_speaker_id)
                        agent = self.agents[next_speaker_id]
                        agent.has_speaking_token = True
                        
                        # Get speaking context
                        context = self.orchestrator.get_speaking_context()
                        
                        # Have the agent speak
                        logger.info(f"Agent {agent.name} is speaking...")
                        print(f"\n{agent.name} is speaking...")
                        
                        audio_data, sample_rate, position = await agent.speak(context)
                        
                        if audio_data is not None:
                            # Play the audio
                            await self.audio_player.play_to_position(
                                audio_data, position, sample_rate
                            )
                            
                            # Release speaking token
                            await agent.release_speaking_token()
                            
                            # Small pause between speakers
                            await asyncio.sleep(0.5)
                
                # Sleep to avoid busy-waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Conversation management task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in conversation management: {str(e)}")