#!/usr/bin/env python
"""
Test the RAG system with existing knowledge base.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.rag_multi_agent_manager import RagMultiAgentManager
from src.utils.logging import setup_logger

logger = setup_logger("test_rag_system")


async def test_rag_conversation():
    """Test RAG-enhanced conversation without ingesting new documents."""
    print("\n===== Testing RAG System =====")
    print("Using existing knowledge base only\n")
    
    manager = RagMultiAgentManager()
    
    try:
        # Start the manager
        await manager.start()
        
        # Check if knowledge base has data
        kb_has_data = await manager.check_knowledge_base()
        if kb_has_data:
            print("✅ Knowledge base contains data")
        else:
            print("❌ Knowledge base is empty")
            return
        
        # Test knowledge retrieval
        print("\n🔍 Testing knowledge retrieval...")
        results = await manager.knowledge_base.query("artificial intelligence consciousness", n_results=3)
        if results:
            print(f"Found {len(results)} relevant documents:")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result['text'][:100]}...")
        else:
            print("No relevant documents found")
        
        # Run a simple conversation
        print("\n🤖 Starting RAG-enhanced conversation...")
        
        topic = "artificial intelligence and consciousness"
        turns = 1
        
        # Start the conversation
        await manager.run_conversation(topic, turns)
        
        print("\n✅ RAG system test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    finally:
        try:
            await manager.stop()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_rag_conversation())
