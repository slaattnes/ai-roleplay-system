#!/usr/bin/env python
"""
Stage 4 of the AI Agent Role-Playing System: 
Orchestrated conversation with simplified scheduling.
"""

import asyncio
import sys

from src.orchestrated_agent_manager import OrchestratedAgentManager
from src.utils.logging import setup_logger, system_logger

async def main():
    """Main entry point for Stage 4."""
    print("\n===== AI Agent Role-Playing System: Stage 4 =====")
    print("Orchestrated Conversation with Enhanced Turn-Taking\n")
    
    # Create the orchestrated agent manager
    manager = OrchestratedAgentManager()
    
    try:
        # Start the manager
        print("Starting the agent system with premiere session...")
        await manager.start()
        
        # Display transcript location
        print(f"\nTranscript will be saved to: {manager.transcript_logger.get_transcript_path()}")
        
        # The scheduler will automatically run the premiere session
        # Just wait until all events are completed or user interrupts
        try:
            # Keep the main task running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
    except Exception as e:
        system_logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
    finally:
        # Print transcript location again
        if hasattr(manager, 'transcript_logger'):
            print(f"\nConversation transcript saved to: {manager.transcript_logger.get_transcript_path()}")
        
        # Clean up resources
        print("\nStopping the agent system...")
        await manager.stop()
        print("System stopped.")

if __name__ == "__main__":
    asyncio.run(main())