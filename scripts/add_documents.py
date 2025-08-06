#!/usr/bin/env python3
"""
Add documents to the RAG knowledge base.

This script processes documents and adds them to the vector database.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.rag import KnowledgeBase
from src.knowledge.document_processor import DocumentProcessor
from src.utils.logging import setup_logger

logger = setup_logger("add_documents")


async def add_documents_to_kb(document_directory: str = None):
    """Add documents from directory to knowledge base."""
    try:
        # Use default document directory if not specified
        if not document_directory:
            from src.utils.config import load_config
            main_config = load_config("main")
            document_directory = main_config["knowledge"]["document_path"]
        
        doc_dir = Path(document_directory)
        if not doc_dir.exists():
            logger.error(f"Document directory does not exist: {doc_dir}")
            return False
        
        logger.info(f"Processing documents from: {doc_dir}")
        
        # Initialize knowledge base
        kb = KnowledgeBase(collection_name="agent_knowledge")
        
        # Process all documents in the directory
        documents = DocumentProcessor.process_directory(str(doc_dir))
        
        if not documents:
            logger.warning("No documents found to process")
            return True
        
        logger.info(f"Found {len(documents)} documents to process")
        
        # Add each document to the knowledge base
        for i, (text, metadata, file_path) in enumerate(documents):
            doc_id = f"doc_{i}_{Path(file_path).stem}"
            await kb.add_document(text, doc_id, metadata)
            logger.info(f"Added document {i+1}/{len(documents)}: {Path(file_path).name}")
        
        logger.info(f"Successfully added {len(documents)} documents to the knowledge base")
        
        # Test the knowledge base with a query
        logger.info("Testing knowledge base with sample query...")
        results = await kb.query("artificial intelligence", n_results=3)
        logger.info(f"Sample query returned {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add documents to RAG knowledge base")
    parser.add_argument("--directory", "-d", 
                       help="Directory containing documents (default: from config)")
    
    args = parser.parse_args()
    
    success = await add_documents_to_kb(args.directory)
    
    if success:
        logger.info("Documents added successfully!")
    else:
        logger.error("Failed to add documents!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
