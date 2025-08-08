#!/usr/bin/env python
"""
Stage 4 Test: AI Agent Role-Playing System for testing without audio hardware
Version that works in environments without audio devices (like WSL).
"""

import asyncio
import sys
import os

# Mock the audio player to avoid hardware requirements
class MockMultiChannelPlayer:
    """Mock audio player for testing without hardware."""
    
    def __init__(self):
        print("MockMultiChannelPlayer: Initialized (no audio hardware required)")
    
    async def start(self):
        print("MockMultiChannelPlayer: Started")
    
    async def play_to_position(self, audio_data, position, sample_rate, effects=None):
        print(f"MockMultiChannelPlayer: Playing audio to position {position} (sample_rate: {sample_rate})")
        # Simulate playback time
        duration = len(audio_data) / sample_rate if audio_data is not None else 1.0
        await asyncio.sleep(min(duration, 3.0))  # Cap at 3 seconds for testing
    
    def cleanup(self):
        print("MockMultiChannelPlayer: Cleaned up")

# Replace the real audio player with mock
import src.utils.multi_channel_player
src.utils.multi_channel_player.MultiChannelPlayer = MockMultiChannelPlayer

from src.orchestrated_agent_manager import OrchestratedAgentManager
from src.utils.logging import setup_logger, system_logger

async def main():
    """Main entry point for Stage 4 Test."""
    print("\n===== AI Agent Role-Playing System: Stage 4 Test =====")
    print("Orchestrated Conversation with Transcript Logging (No Audio Hardware Required)\n")
    
    # Create the orchestrated agent manager
    manager = OrchestratedAgentManager()
    
    try:
        # Start the manager
        print("Starting the agent system with premiere session...")
        await manager.start()
        
        # Display transcript location
        print(f"\nTranscript will be saved to: {manager.transcript_logger.get_transcript_path()}")
        
        # Run for a limited time for testing
        print("\nRunning conversation for 60 seconds (press Ctrl+C to stop earlier)...")
        try:
            # Keep the main task running until interrupted or timeout
            await asyncio.wait_for(
                asyncio.create_task(_wait_for_interrupt()),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print("\nTest timeout reached (60 seconds).")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
    except Exception as e:
        system_logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Print transcript location again
        if hasattr(manager, 'transcript_logger'):
            print(f"\nConversation transcript saved to: {manager.transcript_logger.get_transcript_path()}")
        
        # Clean up resources
        print("\nStopping the agent system...")
        await manager.stop()
        print("System stopped.")

async def _wait_for_interrupt():
    """Wait indefinitely for interruption."""
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
