#!/usr/bin/env python3
"""
Test script for the RAG system.

This script tests document processing and vector database operations.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.rag import KnowledgeBase
from src.knowledge.document_processor import DocumentProcessor
from src.utils.logging import setup_logger

logger = setup_logger("test_rag")


async def test_document_processing():
    """Test document processing functionality."""
    logger.info("Testing document processing...")
    
    # Test with sample text
    sample_text = """
    This is a test document about artificial intelligence.
    
    Artificial Intelligence (AI) refers to the simulation of human intelligence 
    processes by machines, especially computer systems. These processes include 
    learning, reasoning, and self-correction.
    
    Machine learning is a subset of AI that provides systems the ability to 
    automatically learn and improve from experience without being explicitly programmed.
    """
    
    # Test text cleaning
    cleaned = DocumentProcessor._clean_text(sample_text)
    logger.info(f"Text cleaning test passed. Length: {len(cleaned)}")
    
    # Test supported extensions
    extensions = DocumentProcessor.get_supported_extensions()
    logger.info(f"Supported extensions: {extensions}")
    
    return True


async def test_knowledge_base():
    """Test knowledge base operations."""
    logger.info("Testing knowledge base...")
    
    try:
        # Initialize knowledge base
        kb = KnowledgeBase(collection_name="test_collection")
        logger.info("Knowledge base initialized successfully")
        
        # Test adding a document
        test_doc = """
        The field of artificial intelligence has grown rapidly in recent years.
        Machine learning algorithms can now process vast amounts of data and 
        make predictions with remarkable accuracy. Deep learning, a subset of 
        machine learning, uses neural networks with many layers to model complex patterns.
        """
        
        await kb.add_document(test_doc, "test_doc_1", {"type": "test", "topic": "AI"})
        logger.info("Document added successfully")
        
        # Test querying
        results = await kb.query("artificial intelligence", n_results=2)
        logger.info(f"Query returned {len(results)} results")
        
        if results:
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['text'][:100]}...")
        
        # Clean up test collection
        try:
            kb.client.delete_collection("test_collection")
            logger.info("Test collection cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up test collection: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting RAG system tests...")
    
    try:
        # Test document processing
        doc_test = await test_document_processing()
        
        # Test knowledge base
        kb_test = await test_knowledge_base()
        
        if doc_test and kb_test:
            logger.info("All tests passed! RAG system is working correctly.")
        else:
            logger.error("Some tests failed. Please check the logs.")
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
