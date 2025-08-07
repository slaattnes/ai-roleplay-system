#!/usr/bin/env python
"""
Stage 4 Ultra Fast: AI Agent Role-Playing System with Maximum Optimizations
- Response caching
- Optimized Gemini client with faster model
- Reduced pauses
- Parallel response generation
"""

import asyncio
import sys

# Import the cached version but modify it to use optimized client
from src.orchestrated_agent_manager_cached import OrchestratedAgentManager
from src.llm.gemini_optimized import OptimizedGeminiClient
from src.utils.logging import setup_logger, system_logger

# Monkey patch the manager to use optimized client
original_init = OrchestratedAgentManager.__init__

def optimized_init(self):
    """Initialize with optimized components."""
    original_init(self)
    # Replace the LLM client with the optimized version
    self.llm_client = OptimizedGeminiClient()
    logger = setup_logger("orchestrated_manager")
    logger.info("Using optimized Gemini client for ultra-fast responses")

OrchestratedAgentManager.__init__ = optimized_init

async def main():
    """Main entry point for Stage 4 Ultra Fast."""
    print("\n===== AI Agent Role-Playing System: Stage 4 Ultra Fast =====")
    print("Maximum Optimizations: Caching + Fast Model + Parallel Generation\n")
    
    # Create the optimized agent manager
    manager = OrchestratedAgentManager()
    
    try:
        # Start the manager
        print("Starting ultra-fast agent system...")
        print("- Response caching enabled")
        print("- Using Gemini 1.5 Flash for faster generation")
        print("- Reduced inter-speaker pauses")
        print("- Parallel response pre-generation")
        await manager.start()
        
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
        print("\nStopping the ultra-fast agent system...")
        await manager.stop()
        print("System stopped.")

if __name__ == "__main__":
    asyncio.run(main())
