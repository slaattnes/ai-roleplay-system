#!/usr/bin/env python3
"""Test script to debug channel assignments."""

import asyncio
import sys
from src.utils.config import load_config

async def main():
    print("=== Channel Assignment Test ===")
    
    # Load agent configs
    agents_config = load_config("agents")
    
    print("\nAgent configurations:")
    for agent_id, config in agents_config.items():
        print(f"{agent_id}: name='{config['name']}', position='{config['position']}', audio_channel={config['audio_channel']}")
    
    # Test position mapping
    print("\nPosition to channel mapping:")
    position_to_channel = {
        "front-left": 0,      # Channel 0 -> Front-left
        "front-right": 1,     # Channel 1 -> Front-right  
        "center": 2,          # No center speaker, map to rear-left
        "rear-left": 2,       # Channel 2 -> Rear-left
        "rear-right": 3,      # Channel 3 -> Rear-right
        "lfe": 5,            # Channel 5 -> LFE/Subwoofer
        "subwoofer": 5,      # Same as LFE
    }
    
    for agent_id, config in agents_config.items():
        position = config['position']
        mapped_channel = position_to_channel.get(position.lower(), 0)
        print(f"{config['name']}: position '{position}' -> channel {mapped_channel}")
    
    print("\nAgent speaking order:")
    agent_ids = list(agents_config.keys())
    for i, agent_id in enumerate(agent_ids):
        config = agents_config[agent_id]
        print(f"{i+1}. {config['name']} (channel {config['audio_channel']})")

if __name__ == "__main__":
    asyncio.run(main())
