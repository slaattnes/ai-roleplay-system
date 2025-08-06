"""Retrieval-Augmented Generation (RAG) for agent knowledge."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

from src.knowledge.document_processor import DocumentProcessor
from src.utils.config import load_config, get_env_var
from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("rag")


class GoogleEmbeddingFunction:
    """Custom embedding function using Google's Generative AI API."""
    
    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        """Initialize with Google API key."""
        genai.configure(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized Google embedding function with model: {model_name}")
    
    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        embeddings = []
        for text in input_texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            except Exception as e:
                logger.error(f"Error generating embedding for text: {e}")
                # Fallback to zero vector if API fails
                embeddings.append([0.0] * 768)  # Standard embedding dimension
        return embeddings


class KnowledgeBase:
    """Knowledge base using ChromaDB for vector storage."""
    
    def __init__(self, collection_name: str = "agent_knowledge"):
        """
        Initialize the knowledge base.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        # Load configuration
        main_config = load_config("main")
        self.vector_db_path = Path(main_config["knowledge"]["vector_db_path"])
        self.document_path = Path(main_config["knowledge"]["document_path"])
        
        # Ensure directories exist
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.document_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.vector_db_path))
        
        # Use Google's embedding API (server-side, no local ML models)
        self.embedding_function = GoogleEmbeddingFunction(
            api_key=get_env_var("GOOGLE_API_KEY")
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    async def add_document(
        self, 
        document_text: str, 
        document_id: str, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a document to the knowledge base.
        
        Args:
            document_text: Text content of the document
            document_id: Unique identifier for the document
            metadata: Optional metadata for the document
        """
        if not metadata:
            metadata = {}
        
        # Split document into chunks for better retrieval
        chunks = self._chunk_document(document_text)
        
        # Add each chunk to the collection
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_metadata = {
                **metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "document_id": document_id
            }
            
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )
        
        logger.info(f"Added document {document_id} with {len(chunks)} chunks")
    
    async def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query_text: Text to find relevant information for
            n_results: Number of results to return
            filter_criteria: Optional filter criteria for the query
            
        Returns:
            List of retrieved documents with text and metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        
        logger.debug(f"Query '{query_text[:30]}...' returned {len(formatted_results)} results")
        return formatted_results
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split a document into overlapping chunks.
        
        Args:
            text: Document text to split
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Adjust chunk boundary to end at a sentence or paragraph
            if end < text_length:
                # Try to find sentence boundaries (period, question mark, exclamation)
                sentence_end = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("\n", start, end)
                )
                
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            # Extract the chunk and add to list
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, with overlap
            start = end - overlap
            
            # Ensure we make progress
            if start >= end:
                start = end + 1
        
        return chunks
    
    async def ingest_directory(self) -> int:
        """
        Ingest all documents from the configured document directory.
        
        Returns:
            Number of documents ingested
        """
        count = 0
        
        # Process all files in the document directory
        document_data = DocumentProcessor.process_directory(str(self.document_path))
        
        # Add each document to the knowledge base
        for text, metadata, file_path in document_data:
            document_id = os.path.basename(file_path)
            await self.add_document(text, document_id, metadata)
            count += 1
        
        logger.info(f"Ingested {count} documents into the knowledge base")
        return count