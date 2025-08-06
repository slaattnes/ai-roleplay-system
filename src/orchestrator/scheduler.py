"""Simplified event scheduler for managing the conversation timeline."""

import asyncio
from typing import Any, Callable, Dict, List

from src.utils.config import load_config
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("scheduler")

class EventScheduler:
    """Simplified scheduler that runs events in sequence."""
    
    def __init__(
        self, 
        on_event: Callable[[str, Dict[str, Any]], None]
    ):
        """
        Initialize the event scheduler.
        
        Args:
            on_event: Callback function when events occur
        """
        # Load schedule configuration
        self.schedule_config = load_config("schedule")
        self.on_event = on_event
        
        self.is_running = False
        self.scheduler_task = None
        
        # Get premiere session events
        self.events = self.schedule_config.get("premiere_session", {}).get("events", [])
        self.next_event_index = 0
        
        logger.info("Event scheduler initialized with premiere session")
    
    async def start(self) -> None:
        """Start the scheduler."""
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Event scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Event scheduler stopped")
    
    async def schedule_immediate_event(self, event_type: str, description: str) -> None:
        """
        Schedule an event to happen immediately.
        
        Args:
            event_type: Type of event to trigger
            description: Description of the event
        """
        event_data = {
            "type": event_type,
            "description": description
        }
        await self.on_event(event_type, event_data)
        logger.info(f"Triggered immediate event: {event_type} - {description}")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that triggers events in sequence."""
        try:
            # Immediately start with the first event
            if self.events:
                first_event = self.events[0]
                event_type = first_event.get("type", "unknown")
                await self.on_event(event_type, first_event)
                self.next_event_index = 1
                logger.info(f"Started premiere session with event: {event_type}")
            
            while self.is_running and self.next_event_index < len(self.events):
                # Get next event
                next_event = self.events[self.next_event_index]
                event_type = next_event.get("type", "unknown")
                duration = next_event.get("duration", 5) * 60  # Convert minutes to seconds
                
                # Wait for the duration of the current event
                short_duration = min(duration, 30)  # Use shorter intervals to check is_running
                elapsed = 0
                
                while elapsed < duration and self.is_running:
                    await asyncio.sleep(short_duration)
                    elapsed += short_duration
                
                if not self.is_running:
                    break
                
                # Trigger the next event
                try:
                    await self.on_event(event_type, next_event)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
                
                # Move to the next event
                self.next_event_index += 1
            
            # All events processed, or scheduler stopped
            if self.next_event_index >= len(self.events):
                logger.info("All scheduled events completed")
                
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in scheduler loop: {str(e)}")