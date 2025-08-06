#!/usr/bin/env python3
"""
Export ChromaDB vector database for transfer to another system.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.knowledge_base import KnowledgeBase
from utils.config import ConfigManager

def export_vector_db(output_path: str, collection_name: str = "ai_roleplay_knowledge"):
    """Export the vector database to a specified path."""
    try:
        # Initialize knowledge base
        config = ConfigManager()
        kb = KnowledgeBase(config.get_rag_config(), collection_name)
        
        # Get the ChromaDB path
        chroma_path = kb.client._settings.persist_directory
        
        if not os.path.exists(chroma_path):
            logging.error(f"No vector database found at {chroma_path}")
            return False
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire ChromaDB directory
        db_export_path = output_dir / "chroma_db"
        if db_export_path.exists():
            shutil.rmtree(db_export_path)
        
        shutil.copytree(chroma_path, db_export_path)
        
        # Create metadata file
        metadata = {
            "collection_name": collection_name,
            "export_timestamp": str(Path().cwd()),
            "documents_count": kb.get_collection_stats()
        }
        
        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Vector database exported to {output_path}")
        logging.info(f"Collection: {collection_name}")
        logging.info(f"Documents: {metadata['documents_count']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to export vector database: {e}")
        return False

def import_vector_db(import_path: str, collection_name: str = "ai_roleplay_knowledge"):
    """Import a vector database from the specified path."""
    try:
        import_dir = Path(import_path)
        
        if not import_dir.exists():
            logging.error(f"Import path does not exist: {import_path}")
            return False
        
        # Check for metadata
        metadata_file = import_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            logging.info(f"Importing collection: {metadata.get('collection_name', 'unknown')}")
            logging.info(f"Documents: {metadata.get('documents_count', 'unknown')}")
        
        # Get target ChromaDB path
        config = ConfigManager()
        rag_config = config.get_rag_config()
        target_path = rag_config.get("vector_db_path", "./data/vector_db")
        
        # Import the database
        source_db = import_dir / "chroma_db"
        if not source_db.exists():
            logging.error(f"No ChromaDB found in import path: {source_db}")
            return False
        
        # Backup existing database if it exists
        target_path = Path(target_path)
        if target_path.exists():
            backup_path = target_path.parent / f"{target_path.name}_backup"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(target_path, backup_path)
            logging.info(f"Backed up existing database to {backup_path}")
        
        # Copy imported database
        shutil.copytree(source_db, target_path)
        
        logging.info(f"Vector database imported to {target_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to import vector database: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Export: python export_vector_db.py export <output_path> [collection_name]")
        print("  Import: python export_vector_db.py import <import_path> [collection_name]")
        sys.exit(1)
    
    command = sys.argv[1]
    path = sys.argv[2]
    collection_name = sys.argv[3] if len(sys.argv) > 3 else "ai_roleplay_knowledge"
    
    if command == "export":
        success = export_vector_db(path, collection_name)
    elif command == "import":
        success = import_vector_db(path, collection_name)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)
