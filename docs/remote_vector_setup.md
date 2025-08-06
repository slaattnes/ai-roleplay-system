# Remote Vector Database Setup Guide

This guide explains how to create vector databases on a GPU computer and use them on your current system.

## Method 1: Process Documents on GPU Computer

### Step 1: Clone Repository on GPU Computer
```bash
git clone <your-repo-url>
cd ai-roleplay-system
pip install -r requirements.txt
```

### Step 2: Batch Process Documents
```bash
# Process all documents in a directory with GPU acceleration
python scripts/batch_process_documents.py ./documents ai_roleplay_knowledge 8 200

# Parameters:
# ./documents          - Directory containing PDFs, EPUBs, etc.
# ai_roleplay_knowledge - Collection name
# 8                    - Number of parallel workers (use more on GPU machine)
# 200                  - Batch size for database insertion
```

### Step 3: Export Vector Database
```bash
# Export the created vector database
python scripts/export_vector_db.py export ./vector_db_export ai_roleplay_knowledge
```

### Step 4: Transfer to Your System
```bash
# Compress for transfer
tar -czf vector_db.tar.gz vector_db_export/

# Transfer via scp, rsync, or cloud storage
scp vector_db.tar.gz user@your-system:/path/to/ai-roleplay-system/
```

### Step 5: Import on Your System
```bash
# Extract and import
tar -xzf vector_db.tar.gz
python scripts/export_vector_db.py import ./vector_db_export ai_roleplay_knowledge
```

## Method 2: Remote Vector Database Server

### Option A: ChromaDB Server Mode
```bash
# On GPU computer - start ChromaDB server
pip install chromadb[server]
chroma run --host 0.0.0.0 --port 8000 --path ./vector_db
```

### Option B: Update Configuration for Remote Database
Create a config file for remote database access:

```toml
# config/remote_rag.toml
[rag]
collection_name = "ai_roleplay_knowledge"
chunk_size = 1000
chunk_overlap = 200
embedding_model = "models/embedding-001"
vector_db_type = "remote_chroma"  # Instead of "chroma"
vector_db_url = "http://gpu-server:8000"  # Remote ChromaDB server
```

## Performance Optimization for GPU Processing

### GPU-Optimized Document Processing
For even better performance on GPU systems, you can modify the embedding function:

```python
# In src/rag/knowledge_base.py - for GPU systems
import chromadb.utils.embedding_functions as embedding_functions

# Use sentence-transformers with GPU
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda"  # Use GPU for embeddings
)
```

## Example Workflow

1. **On GPU Computer:**
   ```bash
   # Clone and setup
   git clone <repo>
   cd ai-roleplay-system
   pip install -r requirements.txt
   
   # Process large document collection
   python scripts/batch_process_documents.py ~/Documents/research_papers ai_knowledge 16 500
   
   # Export for transfer
   python scripts/export_vector_db.py export ./ai_knowledge_export ai_knowledge
   tar -czf ai_knowledge.tar.gz ai_knowledge_export/
   ```

2. **Transfer to Main System:**
   ```bash
   # Via cloud storage, network transfer, etc.
   scp ai_knowledge.tar.gz daniel@main-system:~/ai-roleplay-system/
   ```

3. **On Main System:**
   ```bash
   # Import the vector database
   tar -xzf ai_knowledge.tar.gz
   python scripts/export_vector_db.py import ./ai_knowledge_export ai_knowledge
   
   # Test the imported database
   python scripts/test_rag.py
   ```

## Benefits of This Approach

- **GPU Acceleration**: Faster document processing and embedding generation
- **Scalability**: Process large document collections efficiently
- **Portability**: Vector databases are portable between systems
- **Flexibility**: Can create specialized collections for different topics

## Troubleshooting

### Common Issues:
1. **ChromaDB version compatibility**: Ensure same ChromaDB version on both systems
2. **Collection names**: Use consistent collection names during export/import
3. **Path issues**: Verify all paths exist and are accessible

### Performance Tips:
- Use more workers (`max_workers`) on systems with more CPU cores
- Increase `batch_size` for systems with more RAM
- Use SSD storage for vector databases for faster retrieval
