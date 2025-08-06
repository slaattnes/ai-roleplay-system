#!/usr/bin/env python3
"""Simple audio channel test script."""

import asyncio
import numpy as np
from src.utils.multi_channel_player import MultiChannelPlayer
from src.utils.config import load_config

async def test_channels():
    """Test audio playback on specific channels."""
    print("=== Audio Channel Test ===")
    
    # Initialize audio player
    player = MultiChannelPlayer()
    await player.start()
    
    # Load agent config to see the channel assignments
    agents_config = load_config("agents")
    
    # Create a simple test tone (0.5 second beep at 48kHz)
    sample_rate = 48000
    duration = 0.5
    frequency = 440  # A4 note
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume
    audio_data = (tone * 32767).astype(np.int16)  # Convert to 16-bit
    
    print(f"\nTesting channels with {duration}s tone at {frequency}Hz")
    print("Listen for which physical speaker each agent plays from:\n")
    
    # Test each agent in order
    for agent_id in sorted(agents_config.keys()):
        config = agents_config[agent_id]
        channel = config['audio_channel']
        position = config['position']
        name = config['name']
        
        print(f"{agent_id}: {name}")
        print(f"  Channel: {channel} ({position})")
        print(f"  Playing tone...")
        
        try:
            await player.play_to_channel(audio_data, channel, sample_rate)
            print(f"  ✓ Successfully played on channel {channel}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Pause between tests
        await asyncio.sleep(2)
        print()
    
    # Cleanup
    player.cleanup()
    print("Channel test complete!")

if __name__ == "__main__":
    asyncio.run(test_channels())
