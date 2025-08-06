"""Document processing utilities for the knowledge base."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from src.utils.logging import setup_logger

# Set up module logger
logger = setup_logger("document_processor")


class DocumentProcessor:
    """Processes various document formats for knowledge extraction."""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> Tuple[str, Dict]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            # Extract basic metadata
            metadata = {
                "title": os.path.basename(file_path),
                "pages": len(reader.pages),
                "format": "pdf"
            }
            
            # Try to get document info if available
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", metadata["title"]),
                    "author": reader.metadata.get("/Author", "Unknown"),
                    "creation_date": reader.metadata.get("/CreationDate", "Unknown")
                })
            
            logger.info(f"Processed PDF: {file_path} ({len(reader.pages)} pages)")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return "", {"error": str(e), "format": "pdf"}
    
    @staticmethod
    def extract_from_epub(file_path: str) -> Tuple[str, Dict]:
        """
        Extract text and metadata from an EPUB file.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        try:
            book = epub.read_epub(file_path)
            text = ""
            
            # Extract metadata
            metadata = {
                "title": os.path.basename(file_path),
                "format": "epub"
            }
            
            # Update with book metadata if available
            if book.get_metadata('DC', 'title'):
                metadata["title"] = book.get_metadata('DC', 'title')[0][0]
                
            if book.get_metadata('DC', 'creator'):
                metadata["author"] = book.get_metadata('DC', 'creator')[0][0]
            
            # Extract content from each document
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content()
                soup = BeautifulSoup(content, 'html.parser')
                text += soup.get_text() + "\n\n"
            
            logger.info(f"Processed EPUB: {file_path}")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing EPUB {file_path}: {str(e)}")
            return "", {"error": str(e), "format": "epub"}
    
    @staticmethod
    def extract_from_txt(file_path: str) -> Tuple[str, Dict]:
        """
        Extract text and basic metadata from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            metadata = {
                "title": os.path.basename(file_path),
                "format": "txt"
            }
            
            logger.info(f"Processed text file: {file_path} ({len(text)} chars)")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return "", {"error": str(e), "format": "txt"}
    
    @classmethod
    def process_file(cls, file_path: str) -> Tuple[str, Dict]:
        """
        Process a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        file_path = str(file_path)  # Ensure string type
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return cls.extract_from_pdf(file_path)
        elif ext == '.epub':
            return cls.extract_from_epub(file_path)
        elif ext == '.txt':
            return cls.extract_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext} ({file_path})")
            return "", {"error": "Unsupported format", "format": ext[1:]}
    
    @classmethod
    def process_directory(cls, directory_path: str) -> List[Tuple[str, Dict, str]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of tuples (text, metadata, file_path)
        """
        supported_extensions = ['.pdf', '.epub', '.txt']
        results = []
        
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                logger.error(f"Directory not found: {directory_path}")
                return results
            
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    text, metadata = cls.process_file(file_path)
                    if text:  # Only add if text was successfully extracted
                        results.append((text, metadata, str(file_path)))
            
            logger.info(f"Processed {len(results)} documents from {directory_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return results