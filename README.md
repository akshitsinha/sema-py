# sema-py

A minimal semantic search tool for local text files with incremental indexing and REPL interface.

## Features

- **Semantic search** using Google's EmbeddingGemma-300M model
- **Incremental indexing** with xxHash3-128 change detection (60x faster than SHA-256)
- **REPL interface** with command history and auto-completion
- **Smart chunking** with 800-char chunks, 100-char overlap at sentence boundaries
- **Persistent storage** using ChromaDB vector database
- **Clean UI** with Rich tables, panels, and progress bars

## Installation

Requires Python 3.12+

```bash
# Clone the repository
git clone <repo-url>
cd sema-py

# Install dependencies with uv
uv sync
```

## Usage

```bash
uv run main.py <directory> [options]
```

### Options

- `--extensions`: File extensions to index (default: .txt,.md,.py,.js,.ts)
- `--chunk-size`: Characters per chunk (default: 800)
- `--chunk-overlap`: Overlap between chunks (default: 100)

### Example

```bash
uv run main.py ./docs --extensions .md,.txt
```

## Commands

Once the REPL starts, you can use these commands:

| Command    | Description                              |
| ---------- | ---------------------------------------- |
| `<query>`  | Search for semantic matches              |
| `/help`    | Show available commands                  |
| `/index`   | Reindex directory (skip unchanged files) |
| `/reindex` | Force complete reindex                   |
| `/status`  | Show database statistics                 |
| `/config`  | Show current configuration               |
| `/files`   | List indexed files with chunk counts     |
| `/clear`   | Clear entire database                    |
| `/exit`    | Exit the program                         |
| `Ctrl+C`   | Exit the program                         |

## How It Works

### Indexing

1. **File Discovery**: Scans directory for files matching extensions
2. **Change Detection**: Uses mtime + size + xxHash3-128 to detect changes
3. **Smart Chunking**: Splits text into 800-char chunks with 100-char overlap
4. **Embedding**: Generates 768-dim embeddings with EmbeddingGemma-300M
5. **Storage**: Stores in ChromaDB with metadata (file path, hash, lines, etc.)

### Search

1. **Query Embedding**: Converts search query to 768-dim vector
2. **Similarity Search**: Finds top 5 most similar chunks using cosine similarity
3. **Context Retrieval**: Shows 5 lines of context around matches
4. **Results Display**: Highlights matching lines with file name, line numbers, and scores

### Incremental Indexing

Files are only reindexed if:

- File is new to database
- File modification time or size changed
- File content hash (xxHash3-128) changed

Unchanged files are skipped for fast reindexing.

## Architecture

```
main.py         # CLI entry point with click
repl.py         # REPL interface with prompt-toolkit and Rich
text.py         # Core search engine with ChromaDB and sentence-transformers
chroma_db/      # Persistent vector database storage
```

## Dependencies

- **chromadb** 1.2.2: Vector database with persistent storage
- **sentence-transformers** 5.1.2: EmbeddingGemma-300M model (768 dimensions)
- **xxhash** 3.6.0: Fast file hashing with XXH3-128
- **click** 8.3.0: CLI argument parsing
- **rich** 14.2.0: Terminal UI (tables, panels, progress bars)
- **prompt-toolkit** 3.0.48: REPL with history

## Performance

- **Hashing**: XXH3-128 is ~60x faster than SHA-256 for change detection
- **Chunking**: 800-char chunks with 100-char overlap balances context and granularity
- **Embeddings**: EmbeddingGemma-300M provides 768-dim embeddings (~1GB model)
- **Indexing**: Incremental indexing skips unchanged files automatically

## Example Session

```bash
$ uv run main.py ./docs

3 new · 6 chunks indexed

> neural networks

2 results · 0.15s

┌─ ai_basics.txt ─────────────────────────────── lines 5-8 · score 0.85 ┐
│    5 | Neural networks are computational models inspired by    │
│    6 | the human brain. They consist of layers of              │
│    7 | interconnected nodes that process information.          │
└─────────────────────────────────────────────────────────────────────┘

> /status

  metric    value
  files     3
  chunks    6
  size      0.25 mb

> /exit

goodbye
```

## License

MIT
