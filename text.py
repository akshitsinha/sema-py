import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, cast, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import xxhash
import chromadb
from chromadb.types import Metadata
from sentence_transformers import SentenceTransformer
import pymupdf
import time

from utils import DEVICE


class SemanticSearchEngine:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_name: str = "sentence-transformers/all-minilm-l6-v2",
        collection_name: str = "text_search",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        device: Optional[str] = None,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.device = device if device is not None else DEVICE

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Text file semantic search",
                "model_name": model_name,
            },
        )

        self.model = SentenceTransformer(model_name, device=self.device)

        # Check for model/dimension mismatch
        if self.collection.count() > 0:
            stored_model = self.collection.metadata.get("model_name")
            test_embedding = self.model.encode("test")
            expected_dim = len(test_embedding)

            sample = self.collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                stored_dim = len(sample["embeddings"][0])

                if stored_model and stored_model != model_name:
                    raise ValueError(
                        f"Model mismatch: Collection was created with '{stored_model}', "
                        f"but you're trying to use '{model_name}'. "
                        f"Please clear the database with '/clear' command or delete the '{db_path}' directory."
                    )

                if stored_dim != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch: Collection has {stored_dim}d embeddings, "
                        f"but model '{model_name}' produces {expected_dim}d embeddings. "
                        f"Please clear the database with '/clear' command or delete the '{db_path}' directory."
                    )

    def _compute_file_hash(self, file_path: str) -> str:
        hasher = xxhash.xxh3_128()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(where={"file_path": file_path}, limit=1)
        if results["ids"] and results["metadatas"]:
            return dict(cast(Metadata, results["metadatas"][0]))
        return None

    def _needs_reindex(self, file_path: str) -> bool:
        stored_metadata = self._get_file_metadata(file_path)
        if not stored_metadata:
            return True

        stat = os.stat(file_path)

        # Quick check: mtime and size
        if stat.st_mtime == stored_metadata.get(
            "file_mtime"
        ) and stat.st_size == stored_metadata.get("file_size"):
            return False

        # Hash check
        return self._compute_file_hash(file_path) != stored_metadata.get("file_hash")

    def _delete_file_chunks(self, file_path: str):
        results = self.collection.get(where={"file_path": file_path})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def _chunk_text(self, text: str, file_path: str) -> List[Dict]:
        lines = text.split("\n")
        chunks_with_metadata = []

        current_chunk_lines = []
        current_line_nums = []
        current_length = 0

        for line_num, line in enumerate(lines, start=1):
            line_length = len(line) + 1

            if current_length + line_length > self.chunk_size and current_chunk_lines:
                chunks_with_metadata.append(
                    {
                        "text": "\n".join(current_chunk_lines),
                        "line_start": current_line_nums[0],
                        "line_end": current_line_nums[-1],
                    }
                )

                overlap_lines = max(
                    1,
                    int(
                        self.chunk_overlap / (current_length / len(current_chunk_lines))
                    )
                    if current_length > 0
                    else 1,
                )
                current_chunk_lines = current_chunk_lines[-overlap_lines:] + [line]
                current_line_nums = current_line_nums[-overlap_lines:] + [line_num]
                current_length = sum(
                    len(chunk_line) + 1 for chunk_line in current_chunk_lines
                )
            else:
                current_chunk_lines.append(line)
                current_line_nums.append(line_num)
                current_length += line_length

        if current_chunk_lines:
            chunks_with_metadata.append(
                {
                    "text": "\n".join(current_chunk_lines),
                    "line_start": current_line_nums[0],
                    "line_end": current_line_nums[-1],
                }
            )

        return chunks_with_metadata

    def _read_pdf(self, file_path: str) -> Optional[str]:
        try:
            doc = pymupdf.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception:
            return None

    def _read_file(self, file_path: str) -> Optional[str]:
        if file_path.lower().endswith(".pdf"):
            return self._read_pdf(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception:
                return None

    def _read_file_streaming(self, file_path: str):
        """Read file in a streaming fashion, yielding lines one at a time."""
        if file_path.lower().endswith(".pdf"):
            # Process PDF pages one at a time for memory efficiency
            try:
                doc = pymupdf.open(file_path)
                for page in doc:
                    page_text = page.get_text()
                    if isinstance(page_text, str):
                        # Yield lines from this page
                        for line in page_text.split("\n"):
                            if line.strip():  # Skip empty lines
                                yield line
                doc.close()
            except Exception:
                return
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line.rstrip("\n\r")
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        for line in f:
                            yield line.rstrip("\n\r")
                except Exception:
                    return

    def _index_single_file(self, file_path: str, batch_size: int = 25) -> int:
        """Index a single file using streaming processing to minimize memory usage."""
        line_generator = self._read_file_streaming(file_path)
        if not line_generator:
            return 0

        stat = os.stat(file_path)
        file_hash = self._compute_file_hash(file_path)

        total_chunks = 0
        chunk_batch = []

        # Process chunks in streaming fashion
        for chunk_metadata in self._chunk_text_streaming(line_generator, file_path, batch_size):
            chunk_batch.append(chunk_metadata)

            # When we have a full batch, process it
            if len(chunk_batch) >= batch_size:
                self._process_chunk_batch(chunk_batch, file_path, file_hash, stat, total_chunks)
                total_chunks += len(chunk_batch)
                chunk_batch = []  # Clear the batch to free memory

        # Process remaining chunks
        if chunk_batch:
            self._process_chunk_batch(chunk_batch, file_path, file_hash, stat, total_chunks)
            total_chunks += len(chunk_batch)

        return total_chunks

    def _find_files(self, input_dir: str, extensions: List[str]) -> List[str]:
        input_path = Path(input_dir)
        files = []
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            files.extend(input_path.rglob(f"*{ext}"))
        return [str(f.resolve()) for f in files]

    def index_directory(
        self,
        input_dir: str,
        file_extensions: List[str] = [".txt", ".pdf"],
        force: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, Union[int, float]]:
        start_time = time.time()

        files = self._find_files(input_dir, file_extensions)

        new_files = []
        updated_files = []
        skipped_files = []

        for file_path in files:
            if force:
                if self._get_file_metadata(file_path):
                    self._delete_file_chunks(file_path)
                    updated_files.append(file_path)
                else:
                    new_files.append(file_path)
            else:
                if not self._get_file_metadata(file_path):
                    new_files.append(file_path)
                elif self._needs_reindex(file_path):
                    self._delete_file_chunks(file_path)
                    updated_files.append(file_path)
                else:
                    skipped_files.append(file_path)

        files_to_index = new_files + updated_files
        total_chunks = 0

        if files_to_index:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self._index_single_file, f): f
                    for f in files_to_index
                }
                for future in as_completed(future_to_file):
                    try:
                        chunks = future.result()
                        total_chunks += chunks
                    except Exception as e:
                        file_path = future_to_file[future]
                        print(f"Error indexing {file_path}: {e}")

        elapsed_time = time.time() - start_time

        return {
            "new": len(new_files),
            "updated": len(updated_files),
            "skipped": len(skipped_files),
            "total": len(files),
            "chunks": total_chunks,
            "time": elapsed_time,
        }

    def search(
        self, query: str, n_results: int = 5, directory_filter: Optional[str] = None
    ) -> List[Dict]:
        query_embedding = self.model.encode(query).tolist()

        # Optimize fetch count based on whether filtering is needed
        if directory_filter:
            # When filtering, we need more results to account for filtering
            fetch_count = min(n_results * 5, 50)  # Reduced from 10x to 5x, capped at 50
        else:
            # When no filtering, we can fetch exactly what we need
            fetch_count = n_results

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Normalize directory path for filtering
        dir_path_normalized = None
        if directory_filter:
            dir_path_normalized = str(Path(directory_filter).resolve())

        search_results = []
        for i in range(len(results["ids"][0])):
            if (
                not results["metadatas"]
                or not results["documents"]
                or not results["distances"]
            ):
                continue

            metadata = cast(Metadata, results["metadatas"][0][i])
            document = results["documents"][0][i]
            distance = results["distances"][0][i]

            file_path_str = str(metadata["file_path"])

            # Filter by directory if specified
            if dir_path_normalized:
                if not file_path_str.startswith(dir_path_normalized):
                    continue

            line_start = metadata.get("line_start", 0)
            line_end = metadata.get("line_end", 0)

            search_results.append(
                {
                    "file_path": file_path_str,
                    "line_start": int(line_start)
                    if isinstance(line_start, (int, float))
                    else 0,
                    "line_end": int(line_end)
                    if isinstance(line_end, (int, float))
                    else 0,
                    "score": 1 / (1 + distance),
                    "text": document,
                    "distance": distance,
                }
            )

            # Stop early if we have enough results (optimization for non-filtered searches)
            if not directory_filter and len(search_results) >= n_results:
                break

        return search_results[:n_results]

    def get_context(
        self, file_path: str, line_start: int, line_end: int, context_lines: int = 5
    ) -> Tuple[str, int, int]:
        # Get all chunks for this file from ChromaDB
        results = self.collection.get(
            where={"file_path": file_path}, include=["documents", "metadatas"]
        )

        if not results["ids"] or not results["metadatas"] or not results["documents"]:
            return "", line_start, line_end

        # Find chunks that overlap with our target range
        relevant_chunks = []
        for i, metadata in enumerate(results["metadatas"]):
            if metadata is None:
                continue
            chunk_start = metadata.get("line_start", 0)
            chunk_end = metadata.get("line_end", 0)

            # Ensure we have integers
            if not isinstance(chunk_start, int) or not isinstance(chunk_end, int):
                continue

            # Check if chunk overlaps with extended range (including context)
            extended_start = max(1, line_start - context_lines)
            extended_end = line_end + context_lines

            if chunk_start <= extended_end and chunk_end >= extended_start:
                relevant_chunks.append(
                    {
                        "text": results["documents"][i],
                        "start": chunk_start,
                        "end": chunk_end,
                    }
                )

        if not relevant_chunks:
            return "", line_start, line_end

        # Sort chunks by start line
        relevant_chunks.sort(key=lambda x: x["start"])

        # Combine chunks
        combined_text = "\n".join(chunk["text"] for chunk in relevant_chunks)
        actual_start = relevant_chunks[0]["start"]
        actual_end = relevant_chunks[-1]["end"]

        return combined_text, actual_start, actual_end

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={
                "description": "Text file semantic search",
                "model_name": self.model_name,
            },
        )

    def get_stats(self) -> Dict[str, int | float]:
        all_data = self.collection.get()

        if not all_data["ids"]:
            return {"total_chunks": 0, "total_files": 0, "total_size_mb": 0}

        unique_files = set()
        if all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                if metadata and "file_path" in metadata:
                    unique_files.add(str(metadata["file_path"]))

        db_path = Path(self.db_path)
        total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())

        return {
            "total_chunks": len(all_data["ids"]),
            "total_files": len(unique_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def _chunk_text_streaming(self, line_generator, file_path: str, batch_size: int = 100):
        """Generate chunks from a streaming line generator in batches."""
        current_chunk_lines = []
        current_line_nums = []
        current_length = 0
        line_num = 0

        for line in line_generator:
            line_num += 1
            line_length = len(line) + 1  # +1 for newline

            if current_length + line_length > self.chunk_size and current_chunk_lines:
                # Yield the completed chunk
                chunk_text = "\n".join(current_chunk_lines)
                chunk_metadata = {
                    "text": chunk_text,
                    "line_start": current_line_nums[0],
                    "line_end": current_line_nums[-1],
                }

                yield chunk_metadata

                # Start new chunk with overlap
                overlap_lines = max(
                    1,
                    int(self.chunk_overlap / (current_length / len(current_chunk_lines)))
                    if current_length > 0 else 1
                )
                current_chunk_lines = current_chunk_lines[-overlap_lines:] + [line]
                current_line_nums = current_line_nums[-overlap_lines:] + [line_num]
                current_length = sum(len(chunk_line) + 1 for chunk_line in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_line_nums.append(line_num)
                current_length += line_length

        # Yield the final chunk if it exists
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunk_metadata = {
                "text": chunk_text,
                "line_start": current_line_nums[0],
                "line_end": current_line_nums[-1],
            }
            yield chunk_metadata

    def _process_chunk_batch(self, chunk_batch: List[Dict], file_path: str, file_hash: str, stat, start_chunk_id: int):
        """Process a batch of chunks: generate embeddings and store in database."""
        if not chunk_batch:
            return

        chunk_texts = [chunk["text"] for chunk in chunk_batch]
        embeddings = self.model.encode(chunk_texts, show_progress_bar=False)

        ids = [f"{file_path}_chunk_{start_chunk_id + i}" for i in range(len(chunk_batch))]
        metadatas: List[Metadata] = [
            cast(
                Metadata,
                {
                    "file_path": str(file_path),
                    "file_hash": str(file_hash),
                    "file_mtime": float(stat.st_mtime),
                    "file_size": int(stat.st_size),
                    "chunk_id": int(start_chunk_id + i),
                    "line_start": int(chunk["line_start"]),
                    "line_end": int(chunk["line_end"]),
                },
            )
            for i, chunk in enumerate(chunk_batch)
        ]

        self.collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
