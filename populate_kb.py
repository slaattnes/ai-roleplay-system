#!/usr/bin/env python
"""
Add small documents to the RAG knowledge base for testing.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.knowledge.rag import KnowledgeBase
from src.utils.logging import setup_logger

logger = setup_logger("populate_kb")


async def populate_knowledge_base():
    """Add some test documents to the knowledge base."""
    print("📚 Populating knowledge base with test documents...")
    
    try:
        # Initialize knowledge base
        kb = KnowledgeBase(collection_name="agent_knowledge")
        
        # Add small test documents
        test_docs = [
            {
                "id": "ai_ethics_basics",
                "content": """
                AI Ethics and Fairness
                
                Artificial Intelligence ethics is a critical field that addresses the moral implications 
                of AI systems. Key principles include:
                
                1. Fairness: AI systems should treat all individuals and groups equitably
                2. Transparency: AI decisions should be explainable and understandable
                3. Privacy: Personal data should be protected and used responsibly
                4. Accountability: There should be clear responsibility for AI decisions
                5. Human agency: Humans should maintain meaningful control over AI systems
                
                These principles guide the development of responsible AI that benefits society.
                """,
                "metadata": {"topic": "ai_ethics", "source": "test_doc"}
            },
            {
                "id": "consciousness_theories",
                "content": """
                Theories of Consciousness
                
                Consciousness remains one of the most puzzling phenomena in science and philosophy.
                Several theories attempt to explain consciousness:
                
                1. Global Workspace Theory: Consciousness arises from global information integration
                2. Integrated Information Theory: Consciousness corresponds to integrated information
                3. Higher-Order Thought Theory: Consciousness requires thoughts about thoughts
                4. Attention-Schema Theory: Consciousness is the brain's model of attention
                
                The question of machine consciousness is particularly relevant as AI systems become
                more sophisticated. Can artificial systems truly be conscious, or merely simulate it?
                """,
                "metadata": {"topic": "consciousness", "source": "test_doc"}
            },
            {
                "id": "ai_machine_learning",
                "content": """
                Artificial Intelligence and Machine Learning
                
                Machine learning is a subset of artificial intelligence that enables systems to
                learn and improve from experience. Key concepts include:
                
                - Supervised Learning: Learning from labeled examples
                - Unsupervised Learning: Finding patterns in unlabeled data
                - Reinforcement Learning: Learning through trial and error with rewards
                - Deep Learning: Neural networks with multiple layers
                
                These techniques have revolutionized fields like computer vision, natural language
                processing, and decision-making systems. The rapid advancement of AI raises important
                questions about the future of human-AI interaction and potential machine consciousness.
                """,
                "metadata": {"topic": "machine_learning", "source": "test_doc"}
            }
        ]
        
        # Add documents to knowledge base
        for doc in test_docs:
            await kb.add_document(doc["content"], doc["id"], doc["metadata"])
            print(f"✅ Added document: {doc['id']}")
        
        print(f"\n🎉 Successfully added {len(test_docs)} documents to the knowledge base!")
        
        # Test querying
        print("\n🔍 Testing knowledge retrieval...")
        results = await kb.query("artificial intelligence consciousness", n_results=2)
        
        if results:
            print(f"Found {len(results)} relevant documents:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['text'][:100]}...")
        else:
            print("No results found")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate knowledge base: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = asyncio.run(populate_knowledge_base())
    if success:
        print("\n✅ Knowledge base is ready for testing!")
    else:
        print("\n❌ Failed to populate knowledge base")
        sys.exit(1)
