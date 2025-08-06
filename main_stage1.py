#!/usr/bin/env python
"""
Stage 1 of the AI Agent Role-Playing System: Single speaking agent.
"""

import asyncio
import sys

from src.single_agent import SingleSpeakingAgent
from src.utils.logging import system_logger

async def main():
    """Main entry point for Stage 1."""
    print("\n===== AI Agent Role-Playing System: Stage 1 =====")
    print("Single Speaking Agent Demo\n")
    
    # Create the agent
    agent = SingleSpeakingAgent()
    
    try:
        # Get topic from command line or use default
        topic = sys.argv[1] if len(sys.argv) > 1 else "the nature of consciousness"
        
        print(f"Agent {agent.name} will speak on the topic: {topic}")
        print("Please wait while generating response...\n")
        
        # Have the agent speak on the topic
        await agent.speak_on_topic(topic)
        
        print("\nAgent response complete.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        system_logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
    finally:
        # Clean up resources
        agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())