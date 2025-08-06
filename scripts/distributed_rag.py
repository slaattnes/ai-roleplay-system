#!/usr/bin/env python3
"""
Export/Import vector database for distributed RAG setup.

This script allows you to:
1. Export vector database and documents from a GPU-enabled system
2. Import the exported data on your main system
"""

import asyncio
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge.rag import KnowledgeBase
from src.knowledge.document_processor import DocumentProcessor
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("distributed_rag")


class DistributedRAGManager:
    """Manage export/import of RAG data for distributed processing."""
    
    def __init__(self):
        """Initialize the manager."""
        main_config = load_config("main")
        self.vector_db_path = Path(main_config["knowledge"]["vector_db_path"])
        self.document_path = Path(main_config["knowledge"]["document_path"])
    
    async def export_rag_data(self, export_path: str) -> bool:
        """
        Export RAG data including vector database and documents.
        
        Args:
            export_path: Path where to save the exported tar file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_file = Path(export_path)
            logger.info(f"Exporting RAG data to: {export_file}")
            
            # Create temporary directory for export
            temp_dir = Path("./temp_export")
            temp_dir.mkdir(exist_ok=True)
            
            # Copy vector database
            if self.vector_db_path.exists():
                shutil.copytree(
                    self.vector_db_path, 
                    temp_dir / "vector_db",
                    dirs_exist_ok=True
                )
                logger.info("Vector database copied")
            
            # Copy documents
            if self.document_path.exists():
                shutil.copytree(
                    self.document_path,
                    temp_dir / "documents", 
                    dirs_exist_ok=True
                )
                logger.info("Documents copied")
            
            # Create metadata file
            metadata = {
                "export_type": "rag_data",
                "vector_db_exists": self.vector_db_path.exists(),
                "documents_exist": self.document_path.exists(),
                "vector_db_size": sum(f.stat().st_size for f in self.vector_db_path.rglob('*') if f.is_file()) if self.vector_db_path.exists() else 0,
                "document_count": len(list(self.document_path.rglob('*'))) if self.document_path.exists() else 0
            }
            
            with open(temp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create tar archive
            with tarfile.open(export_file, "w:gz") as tar:
                tar.add(temp_dir, arcname="rag_data")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"RAG data exported successfully to: {export_file}")
            logger.info(f"Export includes {metadata['document_count']} documents and {metadata['vector_db_size']} bytes of vector data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export RAG data: {e}")
            return False
    
    async def import_rag_data(self, import_path: str, overwrite: bool = False) -> bool:
        """
        Import RAG data from exported tar file.
        
        Args:
            import_path: Path to the exported tar file
            overwrite: Whether to overwrite existing data
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            import_file = Path(import_path)
            logger.info(f"Importing RAG data from: {import_file}")
            
            if not import_file.exists():
                logger.error(f"Import file does not exist: {import_file}")
                return False
            
            # Create temporary directory for extraction
            temp_dir = Path("./temp_import")
            temp_dir.mkdir(exist_ok=True)
            
            # Extract tar archive
            with tarfile.open(import_file, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            # Read metadata
            metadata_file = temp_dir / "rag_data" / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                logger.info(f"Import metadata: {metadata}")
            
            # Import vector database
            source_vector_db = temp_dir / "rag_data" / "vector_db"
            if source_vector_db.exists():
                if self.vector_db_path.exists() and not overwrite:
                    logger.warning("Vector database already exists. Use --overwrite to replace it.")
                else:
                    if self.vector_db_path.exists():
                        shutil.rmtree(self.vector_db_path)
                    shutil.copytree(source_vector_db, self.vector_db_path)
                    logger.info("Vector database imported")
            
            # Import documents
            source_documents = temp_dir / "rag_data" / "documents"
            if source_documents.exists():
                if self.document_path.exists() and not overwrite:
                    logger.warning("Documents directory already exists. Use --overwrite to replace it.")
                else:
                    if self.document_path.exists():
                        shutil.rmtree(self.document_path)
                    shutil.copytree(source_documents, self.document_path)
                    logger.info("Documents imported")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            logger.info("RAG data imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import RAG data: {e}")
            return False
    
    async def build_on_gpu_system(self, document_directory: str) -> bool:
        """
        Build vector database on a GPU-enabled system.
        
        Args:
            document_directory: Directory containing documents to process
            
        Returns:
            True if build successful, False otherwise
        """
        try:
            logger.info("Building vector database on GPU system...")
            logger.info(f"Processing documents from: {document_directory}")
            
            # Process documents
            doc_processor = DocumentProcessor()
            documents = doc_processor.process_directory(document_directory)
            
            logger.info(f"Processed {len(documents)} documents")
            
            # Initialize knowledge base
            kb = KnowledgeBase(collection_name="agent_knowledge")
            
            # Add documents to knowledge base
            for i, (text, metadata, file_path) in enumerate(documents):
                doc_id = f"doc_{i}_{Path(file_path).stem}"
                await kb.add_document(text, doc_id, metadata)
                logger.info(f"Added document {i+1}/{len(documents)}: {Path(file_path).name}")
            
            logger.info("Vector database build complete!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector database: {e}")
            return False


async def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed RAG manager")
    parser.add_argument("action", choices=["export", "import", "build"], 
                       help="Action to perform")
    parser.add_argument("--path", required=True, 
                       help="Path for export/import file or document directory for build")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing data during import")
    
    args = parser.parse_args()
    
    manager = DistributedRAGManager()
    
    if args.action == "export":
        success = await manager.export_rag_data(args.path)
    elif args.action == "import":
        success = await manager.import_rag_data(args.path, args.overwrite)
    elif args.action == "build":
        success = await manager.build_on_gpu_system(args.path)
    
    if success:
        logger.info(f"Action '{args.action}' completed successfully!")
    else:
        logger.error(f"Action '{args.action}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
