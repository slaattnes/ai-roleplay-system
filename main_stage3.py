#!/usr/bin/env python
"""
Stage 3 of the AI Agent Role-Playing System: 
RAG-enhanced agents with knowledge-grounded conversations.
"""

import asyncio
import os
import sys
from pathlib import Path

from src.rag_multi_agent_manager import RagMultiAgentManager
from src.utils.config import load_config
from src.utils.logging import setup_logger, system_logger

async def main():
    """Main entry point for Stage 3."""
    print("\n===== AI Agent Role-Playing System: Stage 3 =====")
    print("RAG-Enhanced Knowledge-Grounded Conversations\n")
    
    # Create the RAG-enhanced multi-agent manager
    manager = RagMultiAgentManager()
    
    try:
        # Start the manager
        await manager.start()
        
        # Check if documents directory exists and has files
        main_config = load_config("main")
        docs_path = Path(main_config["knowledge"]["document_path"])
        docs_path.mkdir(parents=True, exist_ok=True)
        
        if not any(docs_path.iterdir()) if docs_path.exists() else False:
            print(f"Notice: No documents found in {docs_path}.")
            print("Please add PDF, EPUB, or TXT files to this directory.")
            print("Continuing without knowledge ingestion...\n")
        else:
            print("Documents found. Skipping knowledge ingestion for now to avoid system issues.")
            print("The agents will operate without RAG-enhanced knowledge.\n")
            # TODO: Implement batched/chunked document processing to avoid crashes
            # await manager.ingest_knowledge()
        
        # Get topic and number of turns from command line or use defaults
        topic = sys.argv[1] if len(sys.argv) > 1 else "the nature of consciousness and AI"
        turns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        
        print(f"\nStarting knowledge-enhanced conversation on topic: {topic}")
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