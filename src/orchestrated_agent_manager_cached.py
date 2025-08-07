"""Multi-agent manager with orchestrator integration and response caching for faster speaker transitions."""

import asyncio
import random
import re
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

from src.llm.gemini import GeminiClient
from src.orchestrator.orchestrator import Orchestrator
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.multi_channel_player import MultiChannelPlayer

# Set up module logger
logger = setup_logger("orchestrated_manager")

class CachedResponse:
    """Container for cached agent responses."""
    
    def __init__(self, text: str, audio_data: np.ndarray, sample_rate: int, position: str, timestamp: float):
        self.text = text
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.position = position
        self.timestamp = timestamp
        self.is_valid = True
    
    def is_expired(self, max_age: float = 30.0) -> bool:
        """Check if the cached response is too old."""
        return time.time() - self.timestamp > max_age

class OrchestratedAgent:
    """Individual agent with orchestrator integration and caching capabilities."""
    
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
        
        # Extract voice parameters with defaults
        voice_params = config.get("voice_params", {})
        self.speaking_rate = voice_params.get("speaking_rate", 1.0)
        self.pitch = voice_params.get("pitch", 0.0)
        
        self.position = config["position"]
        self.audio_channel = config["audio_channel"]
        
        # Audio effects configuration
        self.audio_effects = config.get("audio_effects", {})
        
        # Agent personality traits (randomize for variety)
        self.talkativeness = random.uniform(0.3, 0.8)  # How likely to request to speak
        self.curiosity = random.uniform(0.4, 0.9)      # How likely to ask questions
        self.agreeableness = random.uniform(0.3, 0.8)  # How likely to agree vs disagree
        
        # Determine if this is the moderator
        self.is_moderator = (agent_id == orchestrator.moderator_id or 
                            agent_id == "moderator" or
                            self.role.lower() == "moderator")
        
        if self.is_moderator:
            # Moderators should be more talkative but also more balanced
            self.talkativeness = 0.7
            self.curiosity = 0.6
            self.agreeableness = 0.6
            logger.info(f"Agent {self.name} ({agent_id}) initialized as the moderator")
        
        # Shared components
        self.orchestrator = orchestrator
        self.llm_client = llm_client
        self.tts_client = tts_client
        
        # Agent state
        self.is_speaking = False
        self.has_speaking_token = False
        self.last_spoke_time = 0
        
        # Caching
        self.cached_response: Optional[CachedResponse] = None
        self.is_generating = False  # Prevent multiple simultaneous generations
        
        logger.info(f"Initialized orchestrated agent: {self.name} ({self.role}) - "
                   f"Talkativeness: {self.talkativeness:.2f}, Curiosity: {self.curiosity:.2f}, "
                   f"Agreeableness: {self.agreeableness:.2f}")
    
    async def pre_generate_response(self, context: Dict) -> None:
        """
        Pre-generate and cache a response for this agent.
        
        Args:
            context: Speaking context
        """
        if self.is_generating or (self.cached_response and not self.cached_response.is_expired()):
            return  # Already generating or have valid cache
        
        self.is_generating = True
        try:
            logger.debug(f"Pre-generating response for {self.name}")
            
            # Generate response text
            ssml_response = await self.generate_response(context)
            plain_text = self._extract_plain_text(ssml_response)
            
            # Detect if this voice supports SSML
            is_chirp_hd = "chirp3-hd" in self.voice_name.lower() or "chirp-hd" in self.voice_name.lower()
            text_to_synthesize = plain_text if is_chirp_hd else ssml_response
            use_ssml_flag = not is_chirp_hd
            
            # Synthesize speech
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=text_to_synthesize,
                voice_name=self.voice_name,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
                sample_rate_hertz=48000,
                use_ssml=use_ssml_flag
            )
            
            # Cache the response
            self.cached_response = CachedResponse(
                text=plain_text,
                audio_data=audio_data,
                sample_rate=sample_rate,
                position=self.position,
                timestamp=time.time()
            )
            
            logger.debug(f"Cached response ready for {self.name}")
            
        except Exception as e:
            logger.error(f"Error pre-generating response for {self.name}: {str(e)}")
        finally:
            self.is_generating = False
    
    def invalidate_cache(self) -> None:
        """Invalidate the cached response."""
        if self.cached_response:
            self.cached_response.is_valid = False
            self.cached_response = None
    
    async def request_to_speak(self, reason: str = "want to share a thought") -> None:
        """Request permission to speak from the orchestrator."""
        if random.random() < self.talkativeness:
            await self.orchestrator.request_to_speak(self.agent_id, reason)
    
    async def release_speaking_token(self) -> None:
        """Release the speaking token back to the orchestrator."""
        if self.has_speaking_token:
            await self.orchestrator.release_speaking_token(self.agent_id)
            self.has_speaking_token = False
            self.last_spoke_time = asyncio.get_event_loop().time()
            
            # Invalidate our cache since context has changed
            self.invalidate_cache()
            
            # Sometimes request to speak again later
            if random.random() < self.talkativeness * 0.5:
                delay = random.uniform(2, 5)
                asyncio.create_task(self._delayed_speak_request(delay))
    
    async def _delayed_speak_request(self, delay: float) -> None:
        """Request to speak after a delay."""
        await asyncio.sleep(delay)
        await self.request_to_speak("follow-up comment")
    
    async def speak(self, context: Dict) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """
        Generate and speak a response, using cache if available.
        
        Args:
            context: Speaking context
            
        Returns:
            Tuple of (audio data, sample rate, position) or (None, None, None) if error
        """
        # Debug logging for context
        logger.debug(f"speak method called with context type: {type(context)}, value: {context}")
        
        if not self.has_speaking_token:
            logger.warning(f"Agent {self.name} attempted to speak without token")
            return None, None, None

        try:
            self.is_speaking = True
            
            # Check if we have a valid cached response
            if (self.cached_response and 
                self.cached_response.is_valid and 
                not self.cached_response.is_expired()):
                
                logger.info(f"{self.name} using cached response")
                
                # Add to transcript
                await self.orchestrator.add_transcript(
                    self.agent_id, self.name, self.cached_response.text
                )
                
                # Return cached audio
                self.is_speaking = False
                return (self.cached_response.audio_data, 
                       self.cached_response.sample_rate, 
                       self.cached_response.position)
            
            # No cache available, generate response normally
            logger.info(f"{self.name} generating new response")
            
            # Generate response
            ssml_response = await self.generate_response(context)
            plain_text = self._extract_plain_text(ssml_response)
            
            # Add to transcript
            await self.orchestrator.add_transcript(
                self.agent_id, self.name, plain_text
            )
            
            # Detect if this voice supports SSML
            is_chirp_hd = "chirp3-hd" in self.voice_name.lower() or "chirp-hd" in self.voice_name.lower()
            text_to_synthesize = plain_text if is_chirp_hd else ssml_response
            use_ssml_flag = not is_chirp_hd
            
            # Synthesize speech
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=text_to_synthesize,
                voice_name=self.voice_name,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
                sample_rate_hertz=48000,
                use_ssml=use_ssml_flag
            )
            
            # Return the audio data
            self.is_speaking = False
            return audio_data, sample_rate, self.position
            
        except Exception as e:
            logger.error(f"Error in speak for {self.name}: {str(e)}")
            self.is_speaking = False
            return None, None, None

    async def generate_response(self, context: Dict) -> str:
        """Generate a response based on the context."""
        # Debug logging
        logger.debug(f"generate_response called with context type: {type(context)}, value: {context}")
        
        # Ensure context is a dictionary
        if not isinstance(context, dict):
            logger.error(f"Context is not a dictionary: {type(context)} = {context}")
            # Create a proper context dictionary
            context = {
                "event_type": "general_discussion",
                "topic": "general conversation",
                "description": "General discussion",
                "should_pose_questions": False,
                "agent_names": {}
            }
        
        # Get current event type, topic, and description with more debug logging
        try:
            event_type = context.get("event_type", "")
            logger.debug(f"event_type: {event_type}")
            topic = context.get("topic", "")
            logger.debug(f"topic: {topic}")
            description = context.get("description", "")
            logger.debug(f"description: {description}")
            should_pose_questions = context.get("should_pose_questions", False)
            logger.debug(f"should_pose_questions: {should_pose_questions}")
            agent_names = context.get("agent_names", {})
            logger.debug(f"agent_names: {agent_names}")
        except Exception as e:
            logger.error(f"Error extracting context values: {str(e)}")
            # Provide defaults
            event_type = "general_discussion"
            topic = "general conversation"
            description = "General discussion"
            should_pose_questions = False
            agent_names = {}
        
        # Build system instruction
        system_instruction = self._build_system_instruction(event_type, should_pose_questions)
        
        # Get recent conversation history
        recent_transcripts = self.orchestrator.get_recent_transcripts(min(7, len(self.orchestrator.agent_ids) * 2))
        history_text = ""
        try:
            history_text = "\n".join([
                f"{t['speaker']}: {t['text']}" 
                for t in recent_transcripts
                if isinstance(t, dict) and 'speaker' in t and 'text' in t
            ])
        except Exception as e:
            logger.error(f"Error processing transcripts: {str(e)}, transcripts: {recent_transcripts}")
            history_text = "No previous conversation."
        
        # Build prompt
        prompt = self._build_prompt(
            event_type, topic, history_text, 
            should_pose_questions, agent_names
        )
        
        # Generate response
        logger.info(f"Agent {self.name} generating response for {event_type} on: {topic}")
        response = await self.llm_client.generate_response(
            prompt=prompt,
            system_instruction=system_instruction
        )
        
        # Process response to ensure proper SSML
        return self._process_response_for_ssml(response)
    
    def _build_system_instruction(self, event_type: str, should_pose_questions: bool) -> str:
        """Build system instruction based on event type and role."""
        base_instruction = f"""
        You are {self.name}, a {self.role}. 
        {self.description}
        
        Your responses should be in character and reflect your expertise and persona.
        Keep your responses concise and engaging, about 2-4 sentences in length.
        
        Your personality traits:
        - Talkativeness: {"High" if self.talkativeness > 0.6 else "Moderate" if self.talkativeness > 0.4 else "Low"}
        - Curiosity: {"High" if self.curiosity > 0.6 else "Moderate" if self.curiosity > 0.4 else "Low"}
        - Agreeableness: {"High" if self.agreeableness > 0.6 else "Moderate" if self.agreeableness > 0.4 else "Low"}
        
        Note: Some voices don't support SSML, so format your response as regular text.
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
            Keep your remarks concise but engaging.
            """
        elif event_type == "closing_remarks":
            base_instruction += """
            This is the closing of the session. Summarize key points and thank participants.
            """
        elif should_pose_questions:
            base_instruction += """
            Feel free to pose thoughtful questions to encourage discussion.
            """
        
        return base_instruction.strip()
    
    def _build_prompt(
        self, 
        event_type: str, 
        topic: str, 
        history_text: str, 
        should_pose_questions: bool,
        agent_names: Dict
    ) -> str:
        """Build the prompt for response generation."""
        # agent_names is Dict[str, str] mapping agent_id -> name
        try:
            if isinstance(agent_names, dict):
                # Filter out our own name and create names list
                other_names = [name for agent_id, name in agent_names.items() if agent_id != self.agent_id]
                names_list = ", ".join(other_names) if other_names else "other participants"
            else:
                logger.warning(f"agent_names is not a dict: {type(agent_names)} = {agent_names}")
                names_list = "other participants"
        except Exception as e:
            logger.error(f"Error processing agent_names: {str(e)}")
            names_list = "other participants"
        
        prompt = f"""
        Current Event: {event_type}
        Discussion Topic: {topic}
        Other Participants: {names_list}
        
        Recent Conversation:
        {history_text if history_text else "No previous conversation."}
        
        Generate your response as {self.name}. Keep it natural, engaging, and in character.
        """
        
        if should_pose_questions:
            prompt += "\nConsider asking a thoughtful question to encourage discussion."
        
        return prompt.strip()
    
    def _process_response_for_ssml(self, response: str) -> str:
        """Process response to ensure proper SSML formatting."""
        # For simplicity, just return the text as-is
        return response
    
    def _extract_plain_text(self, ssml: str) -> str:
        """Extract plain text from SSML and clean up formatting artifacts."""
        # Remove all SSML tags
        text = re.sub(r'<[^>]+>', '', ssml)
        
        # Fix common SSML entities
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        
        # Clean up markdown and code formatting artifacts
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold markdown
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove italic markdown
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Remove inline code backticks
        text = re.sub(r'```[^`]*```', '', text)         # Remove code blocks
        text = re.sub(r'#{1,6}\s*', '', text)           # Remove markdown headers
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove markdown links, keep text
        
        # Clean up common artifacts
        text = re.sub(r'\basterix\b', 'asterisk', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhyphen\b', '-', text, flags=re.IGNORECASE)
        text = re.sub(r'\bunderscore\b', '_', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class OrchestratedAgentManager:
    """Manager for agents with orchestrator integration and response caching."""
    
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
        self.caching_task = None
        
        # Caching configuration
        self.enable_caching = True
        self.max_cache_ahead = 2  # Number of likely next speakers to cache
        
        logger.info("Orchestrated agent manager initialized with caching")
    
    async def start(self) -> None:
        """Start the manager and its components."""
        try:
            # Start audio player
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
            
            # Start tasks
            self.is_running = True
            self.conversation_task = asyncio.create_task(self._manage_conversation())
            
            if self.enable_caching:
                self.caching_task = asyncio.create_task(self._manage_caching())
            
            logger.info(f"Orchestrated agent manager started with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error starting orchestrated agent manager: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the manager and its components."""
        self.is_running = False
        
        # Cancel tasks
        if self.conversation_task:
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        if self.caching_task:
            self.caching_task.cancel()
            try:
                await self.caching_task
            except asyncio.CancelledError:
                pass
        
        # Stop orchestrator
        await self.orchestrator.stop()
        
        # Stop audio player
        self.audio_player.cleanup()
        
        logger.info("Orchestrated agent manager stopped")
    
    async def _manage_caching(self) -> None:
        """Background task to manage response caching."""
        try:
            while self.is_running:
                await asyncio.sleep(1)  # Check every second
                
                if not self.orchestrator.current_speaker:
                    continue
                
                # Get likely next speakers
                context = self.orchestrator.get_speaking_context()
                likely_speakers = await self._get_likely_next_speakers()
                
                # Pre-generate responses for likely speakers
                cache_tasks = []
                for agent_id in likely_speakers[:self.max_cache_ahead]:
                    agent = self.agents.get(agent_id)
                    if agent and not agent.is_generating:
                        cache_tasks.append(agent.pre_generate_response(context))
                
                if cache_tasks:
                    await asyncio.gather(*cache_tasks, return_exceptions=True)
                
        except asyncio.CancelledError:
            logger.info("Caching management task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in caching management: {str(e)}")
    
    async def _get_likely_next_speakers(self) -> List[str]:
        """Get a list of agents likely to speak next."""
        current_speaker = self.orchestrator.current_speaker
        
        # Start with agents who have requested to speak
        likely_speakers = [req["agent_id"] for req in self.orchestrator.speak_requests]
        
        # Add agents who haven't spoken recently
        available_agents = [aid for aid in self.orchestrator.agent_ids if aid != current_speaker]
        less_recent_agents = [aid for aid in available_agents if aid not in self.orchestrator.recent_speakers]
        
        # Combine and deduplicate
        for agent_id in less_recent_agents:
            if agent_id not in likely_speakers:
                likely_speakers.append(agent_id)
        
        # Add remaining agents
        for agent_id in available_agents:
            if agent_id not in likely_speakers:
                likely_speakers.append(agent_id)
        
        return likely_speakers
    
    async def _manage_conversation(self) -> None:
        """Main conversation management loop with optimizations."""
        try:
            consecutive_failures = 0
            
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
                        
                        # Have the agent speak (may use cached response)
                        logger.info(f"Agent {agent.name} is speaking...")
                        print(f"\n{agent.name} is speaking...")
                        
                        audio_data, sample_rate, position = await agent.speak(context)
                        
                        if audio_data is not None:
                            # Play the audio with effects if configured
                            await self.audio_player.play_to_position(
                                audio_data, position, sample_rate, agent.audio_effects
                            )
                            
                            # Release speaking token
                            await agent.release_speaking_token()
                            consecutive_failures = 0
                            
                            # Trigger random speak requests
                            await self._trigger_random_speak_requests()
                            
                            # Reduced pause between speakers for faster transitions
                            pause_time = random.uniform(0.1, 0.3)  # Reduced from 0.3-1.0
                            await asyncio.sleep(pause_time)
                        else:
                            # Handle failure
                            consecutive_failures += 1
                            logger.warning(f"Failed to get audio from {agent.name}, consecutive failures: {consecutive_failures}")
                            await agent.release_speaking_token()
                            
                            if consecutive_failures >= 3:
                                logger.error("Too many consecutive failures, changing topics")
                                new_topic = self.orchestrator._get_next_topic()
                                self.orchestrator.state.update_topic(new_topic)
                                logger.info(f"Changed topic to: {new_topic}")
                                consecutive_failures = 0
                                
                                # Invalidate all caches when topic changes
                                for agent in self.agents.values():
                                    agent.invalidate_cache()
                    else:
                        # No valid speaker available, short wait
                        await asyncio.sleep(0.5)
                
                # Sleep to avoid busy-waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Conversation management task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in conversation management: {str(e)}")
    
    async def _trigger_random_speak_requests(self) -> None:
        """Randomly have some agents request to speak."""
        # 30% chance of triggering speak requests
        if random.random() < 0.3:
            # Choose 1-2 random agents
            num_requesters = random.randint(1, min(2, len(self.agents) - 1))
            potential_requesters = [
                agent for agent_id, agent in self.agents.items() 
                if agent_id != self.orchestrator.current_speaker
            ]
            
            if potential_requesters:
                requesters = random.sample(potential_requesters, min(num_requesters, len(potential_requesters)))
                
                for agent in requesters:
                    # Randomize reasons for wanting to speak
                    reasons = [
                        "have a question",
                        "want to add a perspective",
                        "have a relevant point",
                        "want to respond to previous comment",
                        "have a different interpretation"
                    ]
                    reason = random.choice(reasons)
                    await agent.request_to_speak(reason)
