"""Multi-agent manager with orchestrator integration for advanced conversation logic."""

import asyncio
import datetime
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.llm.gemini import GeminiClient
from src.orchestrator.orchestrator import Orchestrator
from src.tts.google_tts import GoogleTTSClient
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.multi_channel_player import MultiChannelPlayer
from src.utils.transcript_logger import TranscriptLogger

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
        
        # Determine if this is the moderator - check both agent_id and orchestrator's moderator_id
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
        
        logger.info(f"Initialized orchestrated agent: {self.name} ({self.role}) - "
                   f"Talkativeness: {self.talkativeness:.2f}, Curiosity: {self.curiosity:.2f}, "
                   f"Agreeableness: {self.agreeableness:.2f}")
    
    async def request_to_speak(self, reason: str = "want to share a thought") -> None:
        """
        Request permission to speak from the orchestrator.
        
        Args:
            reason: Reason for wanting to speak
        """
        # Only request to speak if we're talkative enough
        if random.random() < self.talkativeness:
            await self.orchestrator.request_to_speak(self.agent_id, reason)
    
    async def release_speaking_token(self) -> None:
        """Release the speaking token back to the orchestrator."""
        if self.has_speaking_token:
            await self.orchestrator.release_speaking_token(self.agent_id)
            self.has_speaking_token = False
            self.last_spoke_time = asyncio.get_event_loop().time()
            
            # Sometimes request to speak again later
            if random.random() < self.talkativeness * 0.5:
                # Schedule a future request to speak
                delay = random.uniform(2, 5)
                asyncio.create_task(self._delayed_speak_request(delay))
    
    async def _delayed_speak_request(self, delay: float) -> None:
        """
        Request to speak after a delay.
        
        Args:
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        await self.request_to_speak("follow-up comment")
    
    async def speak(self, context: Dict) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """
        Generate and speak a response based on the given context.
        
        Args:
            context: Speaking context
            
        Returns:
            Tuple of (audio data, sample rate, position) or (None, None, None) if error
        """
        if not self.has_speaking_token:
            logger.warning(f"Agent {self.name} attempted to speak without token")
            return None, None, None

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
            
            # Detect if this voice supports SSML (Chirp HD voices don't)
            is_chirp_hd = "chirp3-hd" in self.voice_name.lower() or "chirp-hd" in self.voice_name.lower()
            
            # Use appropriate text format and SSML setting
            text_to_synthesize = plain_text if is_chirp_hd else ssml_response
            use_ssml_flag = not is_chirp_hd
            
            if is_chirp_hd:
                logger.debug(f"Using plain text for Chirp HD voice: {self.voice_name}")
            else:
                logger.debug(f"Using SSML for standard voice: {self.voice_name}")
            
            # Synthesize speech
            audio_data, sample_rate = await self.tts_client.synthesize_speech(
                text=text_to_synthesize,
                voice_name=self.voice_name,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
                sample_rate_hertz=48000,  # Match the player's sample rate
                use_ssml=use_ssml_flag
            )
            
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
        should_pose_questions = context.get("should_pose_questions", False)
        agent_names = context.get("agent_names", {})
        
        # Build system instruction
        system_instruction = self._build_system_instruction(event_type, should_pose_questions)
        
        # Get recent conversation history
        recent_transcripts = self.orchestrator.get_recent_transcripts(min(7, len(self.orchestrator.agent_ids) * 2))
        history_text = "\n".join([
            f"{t['speaker']}: {t['text']}" 
            for t in recent_transcripts
        ])
        
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
            This is the closing of the session. Summarize key points discussed and offer
            concluding thoughts. Thank the participants.
            """
        elif event_type == "recess":
            base_instruction += """
            This is a brief recess. Engage in lighter, more informal conversation.
            You can briefly reflect on the discussion so far or mention related topics.
            """
        
        # Add question guidance
        if should_pose_questions:
            base_instruction += """
            Consider occasionally posing thoughtful questions to other participants.
            """
        
        return base_instruction
    
    def _build_prompt(
        self, 
        event_type: str, 
        topic: str, 
        history: str,
        should_pose_questions: bool,
        agent_names: Dict[str, str]
    ) -> str:
        """Build prompt based on event type and context."""
        # Remove own name from agent names
        other_agents = {k: v for k, v in agent_names.items() if k != self.agent_id}
        
        prompt = f"As {self.name}, respond to the topic: {topic}\n\n"
        
        # Add conversation history
        if history:
            prompt += f"Recent conversation:\n{history}\n\n"
        else:
            prompt += "This is the start of the conversation.\n\n"
        
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
        
        # Encourage questions based on personality and context
        if should_pose_questions and random.random() < self.curiosity:
            random_agent_name = random.choice(list(other_agents.values())) if other_agents else None
            if random_agent_name:
                prompt += f"\nConsider directing a question to {random_agent_name} about their perspective."
        
        # Determine agreement tendency based on personality
        if history and random.random() > self.agreeableness:
            prompt += "\nYou might respectfully disagree with some points raised by others."
        elif history and random.random() < self.agreeableness:
            prompt += "\nYou might build upon or agree with points raised by others."
        
        # Keep response concise
        prompt += "\n\nKeep your response concise, around 3-4 sentences."
        prompt += "\nSpeak naturally as a human would in conversation. Avoid mentioning formatting elements like 'asterisk', 'underscore', 'backticks', or other code-related terms."
        
        return prompt
    
    def _process_response_for_ssml(self, response: str) -> str:
        """Process LLM response to ensure proper SSML formatting."""
        # For simplicity, just return the text as-is - we're handling SSML in the TTS module
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
        text = re.sub(r'\basterix\b', 'asterisk', text, flags=re.IGNORECASE)  # Fix common mispronunciation
        text = re.sub(r'\bhyphen\b', '-', text, flags=re.IGNORECASE)           # Replace "hyphen" with actual hyphen
        text = re.sub(r'\bunderscore\b', '_', text, flags=re.IGNORECASE)       # Replace "underscore" with actual underscore
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
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
        
        # Create transcript logger
        session_name = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.transcript_logger = TranscriptLogger(session_name)
        
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
            consecutive_failures = 0  # Track failures to avoid infinite loops
            current_topic = ""
            
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
                        
                        # Check if topic changed
                        if context["topic"] != current_topic:
                            current_topic = context["topic"]
                            self.transcript_logger.log_topic_change(current_topic)
                        
                        # Have the agent speak
                        logger.info(f"Agent {agent.name} is speaking...")
                        print(f"\n{agent.name} is speaking...")
                        
                        audio_data, sample_rate, position = await agent.speak(context)
                        
                        if audio_data is not None:
                            # Get the transcript of what was said
                            recent_transcripts = self.orchestrator.get_recent_transcripts(1)
                            if recent_transcripts:
                                # Log to transcript file
                                self.transcript_logger.log_utterance(
                                    speaker_name=agent.name,
                                    text=recent_transcripts[0]["text"],
                                    event_type=context["event_type"],
                                    topic=context["topic"]
                                )
                            
                            # Play the audio with effects if configured
                            await self.audio_player.play_to_position(
                                audio_data, position, sample_rate, agent.audio_effects
                            )
                            
                            # Release speaking token
                            await agent.release_speaking_token()
                            consecutive_failures = 0  # Reset failure counter on success
                            
                            # Randomly have other agents request to speak
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
                                # Force a topic change to break out of potential loops
                                new_topic = self.orchestrator._get_next_topic()
                                self.orchestrator.state.update_topic(new_topic)
                                logger.info(f"Changed topic to: {new_topic}")
                                self.transcript_logger.log_topic_change(new_topic)
                                consecutive_failures = 0
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