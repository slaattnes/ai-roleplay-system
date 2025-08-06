#!/usr/bin/env python3
"""
Quick test to add a single document to the RAG knowledge base.

This script processes a single document to test the RAG system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.rag import KnowledgeBase
from src.utils.logging import setup_logger

logger = setup_logger("quick_rag_test")


async def quick_test():
    """Quick test of RAG system with a single document."""
    try:
        logger.info("Starting quick RAG test...")
        
        # Initialize knowledge base
        kb = KnowledgeBase(collection_name="agent_knowledge")
        logger.info("Knowledge base initialized")
        
        # Add a simple test document
        test_text = """
        Artificial Intelligence and Ethics
        
        AI ethics encompasses fairness, privacy, transparency, and human oversight.
        Key challenges include algorithmic bias, data protection, and ensuring AI decisions
        are explainable and accountable. As AI systems become more powerful, ethical
        considerations become increasingly important for responsible development.
        """
        
        await kb.add_document(test_text, "ethics_doc", {"topic": "AI ethics", "source": "test"})
        logger.info("Document added successfully")
        
        # Test querying
        results = await kb.query("AI ethics and fairness", n_results=2)
        logger.info(f"Query returned {len(results)} results")
        
        if results:
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['text'][:100]}...")
                logger.info(f"Metadata: {result['metadata']}")
        
        logger.info("✅ Quick RAG test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    asyncio.run(quick_test())
