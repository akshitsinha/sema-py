# sema-py

A semantic search engine for local files and images with GPU acceleration and multi-modal retrieval capabilities.

## Demo

### Web Interface

<p align="center">
  <img src="demo/text query demo.png" alt="Text Query Demo" width="800"/>
</p>

<p align="center">
  <img src="demo/semantic search home.png" alt="Semantic Search Home" width="800"/>
</p>

### Terminal Interface

<p align="center">
  <img src="demo/tui.gif" alt="Terminal Interface Demo" width="800"/>
</p>

## Features

- Search text files and images using natural language queries
- GPU-accelerated embedding generation (Apple Silicon and NVIDIA CUDA)
- Incremental indexing with change detection
- Multi-threaded file processing
- Directory-scoped search with global caching
- CLI and web interfaces
- Persistent vector database storage

## Installation

Requires Python 3.12+

```bash
git clone https://github.com/akshitsinha/sema-py
cd sema-py
uv sync
```

## Usage

### CLI REPL Interface

```bash
uv run main.py <directory> [options]
```

**Options:**

- `--extensions`: File extensions to index (default: .txt, .md, .pdf)
- `--chunk-size`: Characters per chunk (default: 800)
- `--chunk-overlap`: Overlap between chunks (default: 100)

**Example:**

```bash
uv run main.py ./documents --extensions .md,.txt,.pdf --chunk-size 500
```

### Web GUI Interface

```bash
uv run gui.py
```

Opens a Gradio web interface on `http://localhost:7860` with:

- Directory browser and loader
- Text search with context display
- Image search (text-to-image and image-to-image)
- Indexing management and statistics

## REPL Commands

### Text Search

| Command    | Description                               |
| ---------- | ----------------------------------------- |
| `<query>`  | Search for semantic matches in text files |
| `/index`   | Index/reindex text files (skip unchanged) |
| `/reindex` | Force complete reindex of text files      |
| `/status`  | Show database statistics                  |
| `/files`   | List indexed files with chunk counts      |
| `/clear`   | Clear text database                       |

### Image Search

| Command            | Description                       |
| ------------------ | --------------------------------- |
| `/isearch <query>` | Search images by text description |
| `/imsearch <path>` | Find similar images by reference  |
| `/iindex`          | Index/reindex images              |
| `/ireindex`        | Force complete reindex of images  |
| `/istatus`         | Show image database statistics    |
| `/ifiles`          | List indexed images               |
| `/iclear`          | Clear image database              |

### General

| Command   | Description                |
| --------- | -------------------------- |
| `/help`   | Show available commands    |
| `/config` | Show current configuration |
| `/exit`   | Exit the program           |
| `Ctrl+C`  | Exit the program           |

## How It Works

**Indexing:**
Files are scanned recursively and split into overlapping chunks. Each chunk is converted to an embedding vector using EmbeddingGemma or all-MiniLM-L6-v2 (text) or CLIP (images), then stored in ChromaDB. Files are only reindexed when their content changes, detected via hash comparison.

**Search:**
Your query is converted to an embedding vector and compared against stored vectors using cosine similarity. The most similar chunks are retrieved and merged to provide context. Directory filtering allows scoping results while keeping a global cache.

**GPU Support:**
The system automatically detects and uses available GPUs (Apple Silicon or NVIDIA) for faster embedding generation. Falls back to CPU if no GPU is available.

## Example Usage

### Text Search

```bash
$ uv run main.py ./documents

Using device: mps

3 new · 12 chunks indexed
completed in 2.34s

> machine learning fundamentals

3 results · 0.18s

┌─ ml_guide.pdf ──────────────────────────────── lines 12-15 · score 0.87 ┐
│   12 | Machine learning is a subset of artificial intelligence    │
│   13 | that enables systems to learn and improve from experience  │
│   14 | without being explicitly programmed. The core principle    │
│   15 | involves training models on data to make predictions.      │
└────────────────────────────────────────────────────────────────────────┘

> /status

  files     8
  chunks    142
  size      2.5 mb

> /exit
```

### Image Search

```bash
> /isearch sunset over mountains

5 results · 0.22s

┌─ vacation_2024/IMG_5432.jpg ──────────────── score 0.92 ┐
│ /Users/docs/photos/vacation_2024/IMG_5432.jpg           │
│ 1920x1080                                                │
└──────────────────────────────────────────────────────────┘

> /imsearch reference_image.jpg

Finding similar images...

4 results · 0.19s
```

## License

MIT
