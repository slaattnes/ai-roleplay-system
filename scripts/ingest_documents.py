#!/usr/bin/env python3
"""
Enhanced document ingestion script for RAG system.

This script processes various document formats (PDF, EPUB, TXT) and ingests them
into the ChromaDB vector database for Retrieval-Augmented Generation.

Usage:
    python scripts/ingest_documents.py [--documents-dir PATH] [--force-reload]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.rag import KnowledgeBase
from src.knowledge.document_processor import DocumentProcessor
from src.utils.logging import setup_logger

logger = setup_logger("ingest")


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--documents-dir", 
        type=str, 
        default="./data/documents",
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--force-reload", 
        action="store_true",
        help="Force reload all documents (clear existing data)"
    )
    parser.add_argument(
        "--collection-name", 
        type=str, 
        default="agent_knowledge",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Size of text chunks for processing"
    )
    
    args = parser.parse_args()
    
    documents_dir = Path(args.documents_dir)
    
    if not documents_dir.exists():
        logger.error(f"Documents directory does not exist: {documents_dir}")
        documents_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created documents directory: {documents_dir}")
        return
    
    logger.info(f"Ingesting documents from: {documents_dir}")
    logger.info(f"Collection name: {args.collection_name}")
    
    try:
        # Initialize knowledge base
        kb = KnowledgeBase(collection_name=args.collection_name)
        
        if args.force_reload:
            logger.info("Force reload enabled - clearing existing collection")
            try:
                kb.client.delete_collection(args.collection_name)
                logger.info("Cleared existing collection")
            except Exception as e:
                logger.warning(f"Could not clear collection (may not exist): {e}")
            
            # Recreate collection
            kb.collection = kb.client.create_collection(
                name=args.collection_name,
                embedding_function=kb.embedding_function
            )
        
        # Process all documents in the directory
        logger.info("Processing documents...")
        document_data = DocumentProcessor.process_directory(str(documents_dir))
        
        if not document_data:
            logger.warning("No documents found to process")
            return
        
        logger.info(f"Found {len(document_data)} documents to ingest")
        
        # Ingest each document
        ingested_count = 0
        for text, metadata, file_path in document_data:
            try:
                document_id = Path(file_path).stem  # Use filename without extension
                logger.info(f"Ingesting: {file_path}")
                
                # Add file path to metadata
                metadata["file_path"] = file_path
                metadata["file_size"] = len(text)
                
                await kb.add_document(text, document_id, metadata)
                ingested_count += 1
                
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
        
        logger.info(f"Successfully ingested {ingested_count}/{len(document_data)} documents")
        
        # Test the knowledge base with a simple query
        logger.info("Testing knowledge base with sample query...")
        results = await kb.query("artificial intelligence", n_results=3)
        
        if results:
            logger.info(f"Test query returned {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                logger.info(f"Result {i+1}: {result['text'][:100]}...")
        else:
            logger.warning("Test query returned no results")
        
        logger.info("Document ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
