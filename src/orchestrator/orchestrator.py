"""Orchestrator for managing the multi-agent conversation."""

import asyncio
import random
from typing import Dict, List, Optional, Set

from src.orchestrator.scheduler import EventScheduler
from src.orchestrator.state import ConversationState
from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("orchestrator")

class Orchestrator:
    """Central orchestrator that manages the conversation flow."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        # Load configurations
        self.main_config = load_config("main")
        self.agents_config = load_config("agents")
        
        # Conversation state
        self.state = ConversationState()
        
        # Event scheduler
        self.scheduler = EventScheduler(on_event=self._handle_scheduled_event)
        
        # Topic management
        self.topics = [
            "The Nature of Consciousness",
            "AI Ethics and Responsibility",
            "The Future of Human-Machine Collaboration",
            "Philosophical Implications of Artificial Intelligence",
            "The Boundaries Between Human and Machine Cognition",
            "Creativity and Artificial Intelligence",
            "The Role of Emotion in Intelligence",
            "Free Will and Determinism in AI Systems",
            "The Language of Machine Consciousness",
            "The Subjectivity of a Stone",
            "Panpsychist Ethics",
            "The Forest as a Single Mind",
            "The Intersubjective Nature of Reality",
            "The Role of Language in Consciousness",
            "The Ethics of a River",
            "The Consciousness of a Cloud",
            "The Intelligence of a Mountain",
            "The Awakened Proto-Conscious City",
            "The Collective Mind of the Internet",
            "Consciousness as a Flavor"
        ]
        random.shuffle(self.topics)  # Randomize topics
        self.current_topic_index = 0
        
        # Agent management
        self.agent_ids: List[str] = []
        self.moderator_id: Optional[str] = None
        
        # Speaking state
        self.current_speaker: Optional[str] = None
        self.speak_requests: List[Dict] = []
        self.recent_speakers: List[str] = []  # Track recent speakers
        self.consecutive_moderator_turns = 0  # Prevent moderator domination
        
        # Operation flags
        self.is_running = False
        self.allow_interruptions = True  # Default to allowing interruptions for more dynamic conversation
        self.should_pose_questions = True  # Encourage agents to pose questions
        
        logger.info("Orchestrator initialized")
    
    async def start(self) -> None:
        """Start the orchestrator and its background tasks."""
        try:
            # Load agent IDs from config
            self.agent_ids = list(self.agents_config.keys())
            
            # Shuffle agent IDs for more randomness in turn-taking
            random.shuffle(self.agent_ids)
            
            # Find the moderator (specifically looking for agent5, organ_3, moderator, or role)
            self.moderator_id = None
            for agent_id, config in self.agents_config.items():
                # Check for specific agent ID, name patterns, or role
                agent_name = config.get("name", "").lower()
                agent_role = config.get("role", "").lower()
                
                if (agent_id == "agent5" or 
                    agent_id == "organ_3" or 
                    agent_id == "moderator" or 
                    "organ" in agent_name or
                    agent_role == "moderator"):
                    self.moderator_id = agent_id
                    break
            
            # If no specific moderator found, default to first agent
            if not self.moderator_id and self.agent_ids:
                self.moderator_id = self.agent_ids[0]
                
            logger.info(f"Identified moderator: {self.moderator_id}")
            
            # Start the scheduler
            await self.scheduler.start()
            
            # Set initial state
            self.state.update_topic(self._get_next_topic())
            
            self.is_running = True
            logger.info("Orchestrator started")
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the orchestrator and its background tasks."""
        self.is_running = False
        
        # Stop the scheduler
        await self.scheduler.stop()
        
        logger.info("Orchestrator stopped")
    
    async def get_next_speaker(self) -> Optional[str]:
        """
        Get the next agent that should speak.
        
        Returns:
            Agent ID of the next speaker or None
        """
        # For the first speaker, always choose the moderator
        if not self.state.transcripts:  # No transcripts yet means first speaker
            logger.info(f"First speaker - selecting moderator: {self.moderator_id}")
            return self.moderator_id
            
        # If there are pending speak requests, honor the oldest one
        if self.speak_requests:
            request = self.speak_requests.pop(0)
            agent_id = request.get("agent_id")
            reason = request.get("reason", "")
            
            logger.info(f"Granting speak request to {agent_id}: {reason}")
            return agent_id
        
        # Get a list of agents, excluding the current speaker and avoiding too many consecutive turns
        available_agents = [aid for aid in self.agent_ids if aid != self.current_speaker]
        
        # If the moderator has spoken too many times in a row, exclude them temporarily
        if (self.consecutive_moderator_turns >= 2 and 
            self.moderator_id in available_agents):
            available_agents.remove(self.moderator_id)
            logger.debug(f"Temporarily excluding moderator after {self.consecutive_moderator_turns} consecutive turns")
        
        # If all recent speakers have spoken, reset the tracking
        if len(self.recent_speakers) >= len(self.agent_ids) - 1:
            self.recent_speakers = []
        
        # Prioritize agents who haven't spoken recently
        less_recent_agents = [aid for aid in available_agents if aid not in self.recent_speakers]
        
        if less_recent_agents:
            # Choose randomly from less recent speakers
            next_speaker = random.choice(less_recent_agents)
        else:
            # All have spoken recently, so choose randomly
            next_speaker = random.choice(available_agents) if available_agents else None
        
        # Update the tracking of recent speakers
        if next_speaker:
            self.recent_speakers.append(next_speaker)
            
            # Track consecutive moderator turns
            if next_speaker == self.moderator_id:
                self.consecutive_moderator_turns += 1
            else:
                self.consecutive_moderator_turns = 0
                
        return next_speaker
    
    async def request_to_speak(self, agent_id: str, reason: str) -> None:
        """
        Register a request from an agent to speak.
        
        Args:
            agent_id: ID of the requesting agent
            reason: Reason for the request
        """
        if agent_id not in self.agent_ids:
            logger.warning(f"Speak request from unknown agent: {agent_id}")
            return
        
        # Add to request queue
        self.speak_requests.append({
            "agent_id": agent_id,
            "reason": reason,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        logger.info(f"Agent {agent_id} requested to speak: {reason}")
    
    async def release_speaking_token(self, agent_id: str) -> None:
        """
        Handle an agent releasing the speaking token.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id != self.current_speaker:
            logger.warning(f"Agent {agent_id} attempted to release token they don't hold")
            return
        
        self.current_speaker = None
        self.state.set_current_speaker(None)
        logger.info(f"Agent {agent_id} released speaking token")
        
        # Occasionally change topics to keep the conversation fresh
        if random.random() < 0.15:  # 15% chance
            new_topic = self._get_next_topic()
            self.state.update_topic(new_topic)
            logger.info(f"Changing topic to: {new_topic}")
    
    def get_speaking_context(self) -> Dict:
        """
        Get the current speaking context.
        
        Returns:
            Dictionary with context information
        """
        return {
            "event_type": self.state.current_event_type,
            "topic": self.state.current_topic,
            "description": self.state.current_event_description,
            "allow_interruptions": self.allow_interruptions,
            "should_pose_questions": self.should_pose_questions,
            "agent_names": self._get_agent_names()
        }
    
    def _get_agent_names(self) -> Dict[str, str]:
        """
        Get a mapping of agent IDs to names for addressing other agents.
        
        Returns:
            Dictionary mapping agent IDs to names
        """
        names = {}
        for agent_id, config in self.agents_config.items():
            names[agent_id] = config.get("name", agent_id)
        return names
    
    async def add_transcript(self, agent_id: str, speaker_name: str, text: str) -> None:
        """
        Add a transcript to the conversation history.
        
        Args:
            agent_id: ID of the speaking agent
            speaker_name: Display name of the speaker
            text: Transcript text
        """
        self.state.add_transcript(speaker_name, text)
        logger.debug(f"Added transcript from {speaker_name}")
    
    def get_recent_transcripts(self, count: int = 5) -> List[Dict]:
        """
        Get recent transcripts for context.
        
        Args:
            count: Number of transcripts to return
            
        Returns:
            List of recent transcript entries
        """
        return self.state.get_recent_transcripts(count)
    
    def _get_next_topic(self) -> str:
        """
        Get the next topic in rotation.
        
        Returns:
            Topic string
        """
        topic = self.topics[self.current_topic_index]
        self.current_topic_index = (self.current_topic_index + 1) % len(self.topics)
        return topic
    
    async def _handle_scheduled_event(
        self, 
        event_type: str, 
        event_data: Dict
    ) -> None:
        """
        Handle a scheduled event from the EventScheduler.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        description = event_data.get("description", "")
        
        logger.info(f"Scheduled event: {event_type} - {description}")
        
        # Update conversation state
        self.state.update_event(event_type, description)
        
        # For certain events, we might want to update the discussion topic
        if event_type in ["opening_remarks", "keynote", "presentation"]:
            new_topic = self._get_next_topic()
            self.state.update_topic(new_topic)
            logger.info(f"New topic: {new_topic}")
        
        # Set interruption and question policies based on event type
        if event_type in ["open_forum", "roundtable"]:
            self.allow_interruptions = True
            self.should_pose_questions = True
            logger.info("Interruptions allowed, questions encouraged")
        elif event_type in ["structured_discussion"]:
            self.allow_interruptions = False
            self.should_pose_questions = True
            logger.info("Interruptions not allowed, questions encouraged")
        elif event_type in ["keynote", "presentation"]:
            self.allow_interruptions = False
            self.should_pose_questions = False
            logger.info("Interruptions not allowed, questions discouraged")
        else:
            self.allow_interruptions = True
            self.should_pose_questions = random.choice([True, False])
            logger.info(f"Interruptions allowed, questions {'encouraged' if self.should_pose_questions else 'discouraged'}")