# Sema-Py: Local Semantic Search Tool

A CLI tool for semantic search over local text files using vector embeddings.

## Features

- 🔍 **Semantic Search**: Find content by meaning, not just keywords
- ⚡ **Incremental Indexing**: Only re-indexes changed files (using xxHash3-128)
- 📊 **Smart Chunking**: Overlapping chunks at sentence boundaries for better context
- 🎨 **Rich Terminal UI**: Beautiful output with syntax highlighting
- 💾 **Persistent Storage**: ChromaDB vector database for fast retrieval
- 🔄 **Multiple File Types**: Support for .txt, .md, and custom extensions

## Installation

```bash
# Clone or navigate to the project directory
cd sema-py

# Install dependencies
uv sync
```

## Usage

### Index Files

Index all text files in a directory:

```bash
uv run python main.py index --input-dir ./documents
```

With custom file extensions:

```bash
uv run python main.py index --input-dir ./docs --extensions .txt --extensions .md --extensions .py
```

Force re-index all files (ignore change detection):

```bash
uv run python main.py index --input-dir ./docs --force
```

Custom chunking parameters:

```bash
uv run python main.py index --input-dir ./docs \
    --chunk-size 1000 \
    --chunk-overlap 150
```

### Search

Basic search:

```bash
uv run python main.py search "machine learning algorithms"
```

Limit number of results:

```bash
uv run python main.py search "vector embeddings" --top 3
```

Adjust context lines around matches:

```bash
uv run python main.py search "neural networks" --context 10
```

### Check Status

View database statistics:

```bash
uv run python main.py status
```

### Clear Database

Remove all indexed data:

```bash
uv run python main.py clear
```

## How It Works

1. **Indexing**:

   - Files are split into ~800 character chunks with 100 char overlap
   - Chunks split at sentence boundaries to preserve context
   - Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
   - Vectors stored in ChromaDB with metadata (file path, line numbers, hash)

2. **Change Detection**:

   - Fast: Compare file modification time + size (skips 99% of unchanged files)
   - Accurate: Compute xxHash3-128 if mtime/size changed
   - Only re-index files that actually changed

3. **Searching**:
   - Query converted to vector embedding
   - ChromaDB finds most similar chunks using cosine similarity
   - Results displayed with file name, score, and surrounding context
   - Matched lines highlighted in yellow

## Architecture

- `text.py`: Core search engine (indexing, hashing, chunking, search)
- `main.py`: CLI interface (Click framework with Rich output)
- `chroma_db/`: Persistent vector database storage

## Dependencies

- **chromadb** (1.2.2): Vector database
- **sentence-transformers** (5.1.2): Embedding model
- **click** (8.3.0): CLI framework
- **rich** (14.2.0): Terminal formatting
- **xxhash** (3.6.0): Fast hashing for change detection

## Configuration

Default settings:

- Chunk size: 800 characters
- Chunk overlap: 100 characters (12.5%)
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Database path: ./chroma_db
- File extensions: .txt

## Examples

```bash
# Index project documentation
uv run python main.py index --input-dir ./project-docs --extensions .md

# Search for specific concept
uv run python main.py search "database optimization techniques" --top 5

# Check what's indexed
uv run python main.py status

# Re-index everything after changing chunk size
uv run python main.py index --input-dir ./docs --force --chunk-size 1000
```

## Performance

- **Indexing**: ~100 files/second (depends on file size and CPU)
- **Search**: <100ms for typical queries
- **Storage**: ~0.5MB per 1000 chunks
- **Hashing**: xxHash3-128 at ~31 GB/s (60x faster than SHA-256)

## Tips

1. **Large files**: Automatically chunked, no size limit
2. **Multiple directories**: Run index command for each directory
3. **Different file types**: Use `--extensions` multiple times
4. **Semantic search**: Query with natural language, not keywords
5. **Context**: Increase `--context` to see more surrounding lines

## Limitations

- UTF-8 encoding recommended (falls back to latin-1)
- Embedding model limited to 256 tokens per chunk
- Search quality depends on embedding model quality
- No support for binary files (images, PDFs) without preprocessing

## License

MIT
