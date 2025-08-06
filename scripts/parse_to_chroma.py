import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import glob

def parse_documents_to_chroma():
    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./data/chroma")
    
    # Create or get collection
    collection = client.get_or_create_collection(name="roleplay_docs")
    
    documents_dir = "./data/documents"
    all_documents = []
    
    # Load different types of documents
    print("Loading documents...")
    
    # Load text files
    txt_files = glob.glob(os.path.join(documents_dir, "*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded: {txt_file}")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    # Load PDF files
    pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded: {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    # Load EPUB files
    epub_files = glob.glob(os.path.join(documents_dir, "*.epub"))
    for epub_file in epub_files:
        try:
            loader = UnstructuredEPubLoader(epub_file)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded: {epub_file}")
        except Exception as e:
            print(f"Error loading {epub_file}: {e}")
    
    print(f"Total documents loaded: {len(all_documents)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_documents)
    
    print(f"Created {len(splits)} document chunks")
    
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Add to Chroma
    print("Adding chunks to Chroma database...")
    for i, doc in enumerate(splits):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing chunk {i+1}/{len(splits)}")
        
        embedding = embeddings.embed_query(doc.page_content)
        collection.add(
            documents=[doc.page_content],
            embeddings=[embedding],
            metadatas=[doc.metadata],
            ids=[f"doc_{i}"]
        )
    
    print(f"Successfully added {len(splits)} document chunks to Chroma database")

if __name__ == "__main__":
    parse_documents_to_chroma()
