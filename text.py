import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import xxhash
import chromadb
from sentence_transformers import SentenceTransformer


class SemanticSearchEngine:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "text_search",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Text file semantic search"}
        )
        
        self.model = SentenceTransformer(model_name)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute XXH3-128 hash of entire file."""
        hasher = xxhash.xxh3_128()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Get stored metadata for a file if it exists."""
        results = self.collection.get(
            where={"file_path": file_path},
            limit=1
        )
        
        if results['ids']:
            return results['metadatas'][0]
        return None
    
    def _needs_reindex(self, file_path: str) -> bool:
        """Check if file needs re-indexing."""
        stored_metadata = self._get_file_metadata(file_path)
        
        if not stored_metadata:
            return True
        
        stat = os.stat(file_path)
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        
        stored_mtime = stored_metadata.get('file_mtime')
        stored_size = stored_metadata.get('file_size')
        
        # Quick check: mtime and size
        if current_mtime == stored_mtime and current_size == stored_size:
            return False
        
        # Deeper check: hash
        current_hash = self._compute_file_hash(file_path)
        stored_hash = stored_metadata.get('file_hash')
        
        return current_hash != stored_hash
    
    def _delete_file_chunks(self, file_path: str):
        """Delete all chunks for a specific file."""
        results = self.collection.get(
            where={"file_path": file_path}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def _chunk_text(self, text: str, file_path: str) -> List[Dict]:
        """
        Split text into overlapping chunks at sentence boundaries.
        Returns list of dicts with text, line_start, line_end.
        """
        lines = text.split('\n')
        chunks_with_metadata = []
        
        current_chunk_lines = []
        current_line_nums = []
        current_length = 0
        
        for line_num, line in enumerate(lines, start=1):
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > self.chunk_size and current_chunk_lines:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk_lines)
                chunks_with_metadata.append({
                    'text': chunk_text,
                    'line_start': current_line_nums[0],
                    'line_end': current_line_nums[-1]
                })
                
                # Calculate overlap in lines
                if current_length > 0:
                    overlap_lines = max(1, int(self.chunk_overlap / (current_length / len(current_chunk_lines))))
                else:
                    overlap_lines = 1
                
                # Start new chunk with overlap
                current_chunk_lines = current_chunk_lines[-overlap_lines:] + [line]
                current_line_nums = current_line_nums[-overlap_lines:] + [line_num]
                current_length = sum(len(l) + 1 for l in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_line_nums.append(line_num)
                current_length += line_length
        
        # Add last chunk
        if current_chunk_lines:
            chunks_with_metadata.append({
                'text': '\n'.join(current_chunk_lines),
                'line_start': current_line_nums[0],
                'line_end': current_line_nums[-1]
            })
        
        return chunks_with_metadata
    
    def _index_single_file(self, file_path: str) -> int:
        """Index a single file. Returns number of chunks created."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception:
                return 0
        
        if not text.strip():
            return 0
        
        chunks = self._chunk_text(text, file_path)
        
        if not chunks:
            return 0
        
        stat = os.stat(file_path)
        file_hash = self._compute_file_hash(file_path)
        
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        
        ids = [f"{file_path}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_path": file_path,
                "file_hash": file_hash,
                "file_mtime": stat.st_mtime,
                "file_size": stat.st_size,
                "chunk_id": i,
                "line_start": chunk['line_start'],
                "line_end": chunk['line_end'],
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def _find_files(self, input_dir: str, extensions: List[str]) -> List[str]:
        """Find all files with given extensions in directory."""
        input_path = Path(input_dir)
        files = []
        
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            files.extend(input_path.rglob(f'*{ext}'))
        
        return [str(f.resolve()) for f in files]
    
    def index_directory(
        self,
        input_dir: str,
        file_extensions: List[str] = ['.txt'],
        force: bool = False
    ) -> Dict[str, int]:
        """
        Index all files in directory.
        Returns dict with stats: new, updated, skipped, total.
        """
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
                needs_reindex = self._needs_reindex(file_path)
                
                if not self._get_file_metadata(file_path):
                    new_files.append(file_path)
                elif needs_reindex:
                    self._delete_file_chunks(file_path)
                    updated_files.append(file_path)
                else:
                    skipped_files.append(file_path)
        
        files_to_index = new_files + updated_files
        total_chunks = 0
        
        for file_path in files_to_index:
            chunks_created = self._index_single_file(file_path)
            total_chunks += chunks_created
        
        return {
            'new': len(new_files),
            'updated': len(updated_files),
            'skipped': len(skipped_files),
            'total': len(files),
            'chunks': total_chunks
        }
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for semantic matches.
        Returns list of results with file_path, score, text, lines.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'][0]:
            return []
        
        search_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (lower distance = higher similarity)
            score = 1 / (1 + distance)
            
            search_results.append({
                'file_path': metadata['file_path'],
                'line_start': metadata['line_start'],
                'line_end': metadata['line_end'],
                'score': score,
                'text': document,
                'distance': distance
            })
        
        return search_results
    
    def get_context(self, file_path: str, line_start: int, line_end: int, context_lines: int = 5) -> Tuple[str, int, int]:
        """
        Get text from file with surrounding context.
        Returns (text, actual_start_line, actual_end_line).
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return "", line_start, line_end
        
        start = max(0, line_start - context_lines - 1)
        end = min(len(lines), line_end + context_lines)
        
        context_text = ''.join(lines[start:end])
        
        return context_text, start + 1, end
    
    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Text file semantic search"}
        )
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return {
                'total_chunks': 0,
                'total_files': 0,
                'total_size_mb': 0
            }
        
        unique_files = set()
        for metadata in all_data['metadatas']:
            unique_files.add(metadata['file_path'])
        
        db_path = Path(self.db_path)
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        
        return {
            'total_chunks': len(all_data['ids']),
            'total_files': len(unique_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
