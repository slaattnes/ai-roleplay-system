#!/usr/bin/env python3
"""
Reset the vector database.

This script clears the ChromaDB database to fix any schema issues.
"""

import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("reset_vectordb")


def reset_vector_database():
    """Reset the vector database by removing all data."""
    try:
        # Load configuration
        main_config = load_config("main")
        vector_db_path = Path(main_config["knowledge"]["vector_db_path"])
        
        logger.info(f"Resetting vector database at: {vector_db_path}")
        
        # Remove the entire vector database directory
        if vector_db_path.exists():
            shutil.rmtree(vector_db_path)
            logger.info("Vector database directory removed")
        
        # Recreate the directory
        vector_db_path.mkdir(parents=True, exist_ok=True)
        logger.info("Vector database directory recreated")
        
        logger.info("Vector database reset complete!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset vector database: {e}")
        return False


if __name__ == "__main__":
    success = reset_vector_database()
    sys.exit(0 if success else 1)
