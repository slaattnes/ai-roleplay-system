#!/usr/bin/env python3
"""Test all 6 channels to find working ones."""

import asyncio
import numpy as np
from src.utils.multi_channel_player import MultiChannelPlayer

async def test_all_channels():
    """Test all 6 channels individually."""
    print("=== Testing All 6 Channels ===")
    
    # Initialize audio player
    player = MultiChannelPlayer()
    await player.start()
    
    # Create test tone
    sample_rate = 48000
    duration = 5.0  # 1 second tone
    frequency = 440  # A4 note
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
    audio_data = (tone * 32767).astype(np.int16)  # Convert to 16-bit
    
    print(f"Playing {duration}s tone at {frequency}Hz on each channel")
    print("Listen carefully and note which speaker each channel uses:\n")
    
    # Test all 6 channels
    for channel in range(6):
        print(f"Testing Channel {channel}...")
        print("  Playing tone...")
        
        try:
            await player.play_to_channel(audio_data, channel, sample_rate)
            print(f"  ✓ Channel {channel} played successfully")
        except Exception as e:
            print(f"  ✗ Channel {channel} error: {e}")
        
        # Pause between tests
        await asyncio.sleep(1.5)
        print()
    
    # Cleanup
    player.cleanup()
    print("All channel test complete!")
    print("\nPlease report which physical speaker each channel used:")
    print("Channel 0: ?")
    print("Channel 1: ?") 
    print("Channel 2: ?")
    print("Channel 3: ?")
    print("Channel 4: ?")
    print("Channel 5: ?")

if __name__ == "__main__":
    asyncio.run(test_all_channels())
