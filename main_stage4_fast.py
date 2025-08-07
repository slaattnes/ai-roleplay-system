#!/usr/bin/env python
"""
Stage 4 Fast: AI Agent Role-Playing System with Response Caching
Optimized version with reduced time between speakers.
"""

import asyncio
import sys

from src.orchestrated_agent_manager_cached import OrchestratedAgentManager
from src.utils.logging import setup_logger, system_logger

async def main():
    """Main entry point for Stage 4 Fast."""
    print("\n===== AI Agent Role-Playing System: Stage 4 Fast =====")
    print("Orchestrated Conversation with Response Caching for Faster Transitions\n")
    
    # Create the orchestrated agent manager with caching
    manager = OrchestratedAgentManager()
    
    try:
        # Start the manager
        print("Starting the agent system with premiere session and response caching...")
        await manager.start()
        
        # The scheduler will automatically run the premiere session
        # Caching system will pre-generate responses for faster transitions
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
        # Clean up resources
        print("\nStopping the agent system...")
        await manager.stop()
        print("System stopped.")

if __name__ == "__main__":
    asyncio.run(main())
