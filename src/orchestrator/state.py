"""Conversation state management for the orchestrator."""

import time
from typing import Dict, List, Optional

from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("conversation_state")

class ConversationState:
    """Manages the state of the ongoing conversation."""
    
    def __init__(self):
        """Initialize the conversation state."""
        # Current state
        self.current_topic = ""
        self.current_event_type = ""
        self.current_event_description = ""
        
        # Speaking state
        self.current_speaker = None
        self.speaker_start_time = None
        
        # Transcripts
        self.transcripts = []
        self.summary = ""
        
        logger.info("Conversation state initialized")
    
    def update_topic(self, topic: str) -> None:
        """
        Update the current discussion topic.
        
        Args:
            topic: New topic
        """
        self.current_topic = topic
        logger.info(f"Topic updated: {topic}")
    
    def update_event(self, event_type: str, description: str) -> None:
        """
        Update the current event.
        
        Args:
            event_type: Type of event
            description: Event description
        """
        self.current_event_type = event_type
        self.current_event_description = description
        logger.info(f"Event updated: {event_type} - {description}")
    
    def set_current_speaker(self, speaker_id: Optional[str]) -> None:
        """
        Set the current speaker.
        
        Args:
            speaker_id: ID of the speaking agent or None
        """
        self.current_speaker = speaker_id
        
        if speaker_id:
            self.speaker_start_time = time.time()
        else:
            self.speaker_start_time = None
            
        logger.debug(f"Current speaker set to: {speaker_id}")
    
    def add_transcript(self, speaker: str, text: str) -> None:
        """
        Add a transcript entry.
        
        Args:
            speaker: Name of the speaker
            text: Transcript text
        """
        self.transcripts.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time()
        })
        
        # Limit transcript history size
        if len(self.transcripts) > 100:
            self.transcripts = self.transcripts[-100:]
    
    def get_recent_transcripts(self, count: int = 5) -> List[Dict]:
        """
        Get the most recent transcripts.
        
        Args:
            count: Number of transcripts to return
            
        Returns:
            List of recent transcript entries
        """
        return self.transcripts[-count:] if self.transcripts else []
    
    def update_summary(self, summary: str) -> None:
        """
        Update the conversation summary.
        
        Args:
            summary: New summary
        """
        self.summary = summary
        logger.debug("Conversation summary updated")