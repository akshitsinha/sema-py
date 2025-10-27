import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, cast
import xxhash
import chromadb
from chromadb.types import Metadata
from sentence_transformers import SentenceTransformer


class SemanticSearchEngine:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_name: str = "google/embeddinggemma-300m",
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
            metadata={"description": "Text file semantic search", "model_name": model_name}
        )
        
        self.model = SentenceTransformer(model_name)
        
        # Check for model/dimension mismatch
        if self.collection.count() > 0:
            stored_model = self.collection.metadata.get("model_name")
            test_embedding = self.model.encode("test")
            expected_dim = len(test_embedding)
            
            sample = self.collection.get(limit=1, include=["embeddings"])
            if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
                stored_dim = len(sample['embeddings'][0])
                
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
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(where={"file_path": file_path}, limit=1)
        if results['ids'] and results['metadatas']:
            return dict(cast(Metadata, results['metadatas'][0]))
        return None
    
    def _needs_reindex(self, file_path: str) -> bool:
        stored_metadata = self._get_file_metadata(file_path)
        if not stored_metadata:
            return True
        
        stat = os.stat(file_path)
        
        # Quick check: mtime and size
        if (stat.st_mtime == stored_metadata.get('file_mtime') and 
            stat.st_size == stored_metadata.get('file_size')):
            return False
        
        # Hash check
        return self._compute_file_hash(file_path) != stored_metadata.get('file_hash')
    
    def _delete_file_chunks(self, file_path: str):
        results = self.collection.get(where={"file_path": file_path})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def _chunk_text(self, text: str, file_path: str) -> List[Dict]:
        lines = text.split('\n')
        chunks_with_metadata = []
        
        current_chunk_lines = []
        current_line_nums = []
        current_length = 0
        
        for line_num, line in enumerate(lines, start=1):
            line_length = len(line) + 1
            
            if current_length + line_length > self.chunk_size and current_chunk_lines:
                chunks_with_metadata.append({
                    'text': '\n'.join(current_chunk_lines),
                    'line_start': current_line_nums[0],
                    'line_end': current_line_nums[-1]
                })
                
                overlap_lines = max(1, int(self.chunk_overlap / (current_length / len(current_chunk_lines))) if current_length > 0 else 1)
                current_chunk_lines = current_chunk_lines[-overlap_lines:] + [line]
                current_line_nums = current_line_nums[-overlap_lines:] + [line_num]
                current_length = sum(len(chunk_line) + 1 for chunk_line in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_line_nums.append(line_num)
                current_length += line_length
        
        if current_chunk_lines:
            chunks_with_metadata.append({
                'text': '\n'.join(current_chunk_lines),
                'line_start': current_line_nums[0],
                'line_end': current_line_nums[-1]
            })
        
        return chunks_with_metadata
    
    def _index_single_file(self, file_path: str) -> int:
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
        metadatas: List[Metadata] = [
            cast(Metadata, {
                "file_path": str(file_path),
                "file_hash": str(file_hash),
                "file_mtime": float(stat.st_mtime),
                "file_size": int(stat.st_size),
                "chunk_id": int(i),
                "line_start": int(chunk['line_start']),
                "line_end": int(chunk['line_end']),
            })
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
        input_path = Path(input_dir)
        files = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            files.extend(input_path.rglob(f'*{ext}'))
        return [str(f.resolve()) for f in files]
    
    def index_directory(self, input_dir: str, file_extensions: List[str] = ['.txt'], force: bool = False) -> Dict[str, int]:
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
        
        total_chunks = sum(self._index_single_file(f) for f in new_files + updated_files)
        
        return {
            'new': len(new_files),
            'updated': len(updated_files),
            'skipped': len(skipped_files),
            'total': len(files),
            'chunks': total_chunks
        }
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        search_results = []
        for i in range(len(results['ids'][0])):
            if not results['metadatas'] or not results['documents'] or not results['distances']:
                continue
                
            metadata = cast(Metadata, results['metadatas'][0][i])
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            
            line_start = metadata.get('line_start', 0)
            line_end = metadata.get('line_end', 0)
            
            search_results.append({
                'file_path': str(metadata['file_path']),
                'line_start': int(line_start) if isinstance(line_start, (int, float)) else 0,
                'line_end': int(line_end) if isinstance(line_end, (int, float)) else 0,
                'score': 1 / (1 + distance),
                'text': document,
                'distance': distance
            })
        
        return search_results
    
    def get_context(self, file_path: str, line_start: int, line_end: int, context_lines: int = 5) -> Tuple[str, int, int]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return "", line_start, line_end
        
        start = max(0, line_start - context_lines - 1)
        end = min(len(lines), line_end + context_lines)
        return ''.join(lines[start:end]), start + 1, end
    
    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Text file semantic search", "model_name": self.model_name}
        )
    
    def get_stats(self) -> Dict[str, int | float]:
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return {'total_chunks': 0, 'total_files': 0, 'total_size_mb': 0}
        
        unique_files = set()
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                if metadata and 'file_path' in metadata:
                    unique_files.add(str(metadata['file_path']))
        
        db_path = Path(self.db_path)
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        
        return {
            'total_chunks': len(all_data['ids']),
            'total_files': len(unique_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
