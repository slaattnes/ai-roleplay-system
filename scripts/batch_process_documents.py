#!/usr/bin/env python3
"""
Batch process documents and build vector database with GPU acceleration.
Optimized for high-performance document ingestion.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.document_processor import DocumentProcessor
from rag.knowledge_base import KnowledgeBase
from utils.config import ConfigManager

class BatchDocumentProcessor:
    def __init__(self, config_path: str = None):
        """Initialize batch processor with optional config path."""
        self.config = ConfigManager(config_path)
        self.doc_processor = DocumentProcessor()
        self.kb = None
        
    def initialize_knowledge_base(self, collection_name: str = "ai_roleplay_knowledge"):
        """Initialize the knowledge base."""
        try:
            rag_config = self.config.get_rag_config()
            self.kb = KnowledgeBase(rag_config, collection_name)
            logging.info(f"Initialized knowledge base with collection: {collection_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    def find_documents(self, directory: str, extensions: List[str] = None) -> List[Path]:
        """Find all supported documents in directory."""
        if extensions is None:
            extensions = ['.pdf', '.epub', '.txt', '.docx', '.md']
        
        documents = []
        search_dir = Path(directory)
        
        if not search_dir.exists():
            logging.warning(f"Directory does not exist: {directory}")
            return documents
        
        for ext in extensions:
            documents.extend(search_dir.rglob(f"*{ext}"))
        
        logging.info(f"Found {len(documents)} documents in {directory}")
        return documents
    
    def process_single_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document and return results."""
        try:
            start_time = time.time()
            
            # Extract text
            text = self.doc_processor.extract_text(file_path)
            if not text:
                return {
                    "file": str(file_path),
                    "status": "failed",
                    "error": "No text extracted",
                    "processing_time": 0
                }
            
            # Clean text
            cleaned_text = self.doc_processor.clean_text(text)
            
            # Create chunks
            chunks = self.doc_processor.create_chunks(cleaned_text)
            
            processing_time = time.time() - start_time
            
            return {
                "file": str(file_path),
                "status": "success",
                "text_length": len(text),
                "cleaned_length": len(cleaned_text),
                "chunks": len(chunks),
                "chunk_data": chunks,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "file": str(file_path),
                "status": "failed",
                "error": str(e),
                "processing_time": 0
            }
    
    def batch_process_documents(self, 
                              documents: List[Path], 
                              max_workers: int = 4,
                              batch_size: int = 100) -> Dict[str, Any]:
        """Process documents in parallel batches."""
        results = {
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_processing_time": 0,
            "failed_files": [],
            "success_files": []
        }
        
        start_time = time.time()
        
        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_document, doc): doc 
                for doc in documents
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(documents), desc="Processing documents") as pbar:
                batch_chunks = []
                batch_metadatas = []
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        
                        if result["status"] == "success":
                            results["processed"] += 1
                            results["total_chunks"] += result["chunks"]
                            results["total_processing_time"] += result["processing_time"]
                            results["success_files"].append(result["file"])
                            
                            # Add chunks to batch for database insertion
                            for i, chunk in enumerate(result["chunk_data"]):
                                batch_chunks.append(chunk)
                                batch_metadatas.append({
                                    "source": str(file_path),
                                    "chunk_id": i,
                                    "total_chunks": result["chunks"],
                                    "file_size": result["text_length"]
                                })
                            
                            # Insert batch when it reaches batch_size
                            if len(batch_chunks) >= batch_size:
                                self._insert_batch(batch_chunks, batch_metadatas)
                                batch_chunks = []
                                batch_metadatas = []
                        
                        else:
                            results["failed"] += 1
                            results["failed_files"].append({
                                "file": result["file"],
                                "error": result.get("error", "Unknown error")
                            })
                            
                    except Exception as e:
                        results["failed"] += 1
                        results["failed_files"].append({
                            "file": str(file_path),
                            "error": str(e)
                        })
                    
                    pbar.update(1)
                
                # Insert remaining chunks
                if batch_chunks:
                    self._insert_batch(batch_chunks, batch_metadatas)
        
        results["total_time"] = time.time() - start_time
        return results
    
    def _insert_batch(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        """Insert a batch of chunks into the knowledge base."""
        try:
            if self.kb:
                self.kb.add_documents(chunks, metadatas)
                logging.debug(f"Inserted batch of {len(chunks)} chunks")
        except Exception as e:
            logging.error(f"Failed to insert batch: {e}")
    
    def process_directory(self, 
                         directory: str, 
                         collection_name: str = "ai_roleplay_knowledge",
                         max_workers: int = 4,
                         batch_size: int = 100) -> bool:
        """Process all documents in a directory."""
        try:
            # Initialize knowledge base
            if not self.initialize_knowledge_base(collection_name):
                return False
            
            # Find documents
            documents = self.find_documents(directory)
            if not documents:
                logging.warning("No documents found to process")
                return True
            
            # Process documents
            logging.info(f"Starting batch processing of {len(documents)} documents...")
            results = self.batch_process_documents(documents, max_workers, batch_size)
            
            # Report results
            logging.info("=" * 60)
            logging.info("BATCH PROCESSING COMPLETE")
            logging.info("=" * 60)
            logging.info(f"Total documents: {len(documents)}")
            logging.info(f"Successfully processed: {results['processed']}")
            logging.info(f"Failed: {results['failed']}")
            logging.info(f"Total chunks created: {results['total_chunks']}")
            logging.info(f"Total processing time: {results['total_processing_time']:.2f}s")
            logging.info(f"Total time: {results['total_time']:.2f}s")
            logging.info(f"Average time per document: {results['total_time']/len(documents):.2f}s")
            
            if results['failed_files']:
                logging.warning("Failed files:")
                for failed in results['failed_files']:
                    logging.warning(f"  {failed['file']}: {failed['error']}")
            
            # Get final collection stats
            if self.kb:
                stats = self.kb.get_collection_stats()
                logging.info(f"Final collection size: {stats} documents")
            
            return True
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            return False

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python batch_process_documents.py <documents_directory> [collection_name] [max_workers] [batch_size]")
        print("Example: python batch_process_documents.py ./documents ai_knowledge 8 200")
        sys.exit(1)
    
    directory = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "ai_roleplay_knowledge"
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    processor = BatchDocumentProcessor()
    success = processor.process_directory(
        directory=directory,
        collection_name=collection_name,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
