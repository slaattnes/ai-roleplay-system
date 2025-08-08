"""Transcript logger for recording agent conversations."""

import datetime
import os
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("transcript_logger")

class TranscriptLogger:
    """Records agent conversations to transcript files."""
    
    def __init__(self, session_name: Optional[str] = None):
        """
        Initialize the transcript logger.
        
        Args:
            session_name: Optional name for the session
        """
        # Load configuration
        self.main_config = load_config("main")
        
        # Set up transcript directory
        self.transcript_dir = Path(self.main_config["system"]["transcripts_directory"])
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session name if not provided
        self.session_name = session_name or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create transcript file
        self.transcript_file = self.transcript_dir / f"{self.session_name}.md"
        
        # Initialize transcript
        self._initialize_transcript()
        
        logger.info(f"Transcript logger initialized for session '{self.session_name}'")
        logger.info(f"Transcript file: {self.transcript_file}")
    
    def _initialize_transcript(self) -> None:
        """Initialize the transcript file with header."""
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"# AI Agent Conversation: {self.session_name}\n\n")
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Transcript\n\n")
    
    def log_utterance(self, speaker_name: str, text: str, event_type: str = "", topic: str = "") -> None:
        """
        Log an utterance to the transcript file.
        
        Args:
            speaker_name: Name of the speaking agent
            text: Text spoken by the agent
            event_type: Current event type
            topic: Current topic
        """
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        try:
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                # Add event type and topic change markers if provided
                if event_type:
                    f.write(f"\n### Event: {event_type}\n\n")
                
                if topic:
                    f.write(f"\n**Topic:** {topic}\n\n")
                
                # Write the utterance with timestamp
                f.write(f"**[{timestamp}] {speaker_name}:** {text}\n\n")
                
            logger.debug(f"Logged utterance from {speaker_name}")
            
        except Exception as e:
            logger.error(f"Error logging utterance: {str(e)}")
    
    def log_event(self, event_type: str, description: str) -> None:
        """
        Log a system event to the transcript.
        
        Args:
            event_type: Type of event
            description: Event description
        """
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        try:
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(f"\n### [{timestamp}] Event: {event_type}\n")
                f.write(f"*{description}*\n\n")
                
            logger.debug(f"Logged event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
    
    def log_topic_change(self, topic: str) -> None:
        """
        Log a topic change to the transcript.
        
        Args:
            topic: New topic
        """
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        try:
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(f"\n### [{timestamp}] Topic Change\n")
                f.write(f"**New Topic:** {topic}\n\n")
                
            logger.debug(f"Logged topic change: {topic}")
            
        except Exception as e:
            logger.error(f"Error logging topic change: {str(e)}")
    
    def get_transcript_path(self) -> str:
        """
        Get the path to the transcript file.
        
        Returns:
            Path to the transcript file
        """
        return str(self.transcript_file)