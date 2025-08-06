#!/usr/bin/env python
"""
Stage 2 of the AI Agent Role-Playing System: 
Multiple speaking agents with basic turn-taking.
"""

import asyncio
import sys

from src.multi_agent_manager import MultiAgentManager
from src.utils.logging import system_logger

async def main():
    """Main entry point for Stage 2."""
    print("\n===== AI Agent Role-Playing System: Stage 2 =====")
    print("Multiple Speaking Agents with Turn-Taking\n")
    
    # Create the multi-agent manager
    manager = MultiAgentManager()
    
    try:
        # Start the manager
        await manager.start()
        
        # Get topic and number of turns from command line or use defaults
        topic = sys.argv[1] if len(sys.argv) > 1 else "the future of AI and human collaboration"
        turns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        
        print(f"Starting conversation on topic: {topic}")
        print(f"Each agent will speak {turns} times\n")
        
        # Run the conversation
        await manager.run_conversation(topic, turns)
        
        print("\nConversation complete.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        system_logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
    finally:
        # Clean up resources
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())