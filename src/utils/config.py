"""Configuration utilities for the AI agent role-playing system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import toml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define configuration directory
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

def load_config(config_name: str) -> Dict[str, Any]:
    """Load a configuration file from the config directory."""
    config_path = CONFIG_DIR / f"{config_name}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return toml.load(config_path)

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get an environment variable with optional default value."""
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value