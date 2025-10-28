import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, cast
import xxhash
import chromadb
from chromadb.types import Metadata
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage


class ImageSearchEngine:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_name: str = "sentence-transformers/clip-ViT-B-32",
        collection_name: str = "image_search",
    ):
        self.db_path = db_path
        self.model_name = model_name
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Image semantic search", "model_name": model_name}
        )
        
        self.model = SentenceTransformer(model_name, tokenizer_kwargs={'use_fast': True})
        
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
                        f"Please clear the database with '/iclear' command."
                    )
                
                if stored_dim != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch: Collection has {stored_dim}d embeddings, "
                        f"but model '{model_name}' produces {expected_dim}d embeddings. "
                        f"Please clear the database with '/iclear' command."
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
        
        if (stat.st_mtime == stored_metadata.get('file_mtime') and 
            stat.st_size == stored_metadata.get('file_size')):
            return False
        
        return self._compute_file_hash(file_path) != stored_metadata.get('file_hash')
    
    def _delete_file_chunks(self, file_path: str):
        results = self.collection.get(where={"file_path": file_path})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def _extract_names_from_filename(self, filename: str) -> List[str]:
        """Extract potential names from filename."""
        name_part = Path(filename).stem
        
        # Remove common prefixes
        name_part = re.sub(r'^(IMG|DSC|PHOTO|PIC|IMAGE)[-_]?', '', name_part, flags=re.IGNORECASE)
        
        # Remove trailing numbers
        name_part = re.sub(r'\d+$', '', name_part)
        
        # Split on common separators and camelCase
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', name_part)
        parts = re.split(r'[-_\s]+', parts)
        
        # Filter out common words and short parts
        common_words = {'at', 'in', 'the', 'and', 'or', 'with', 'photo', 'image', 'pic'}
        names = [p.lower() for p in parts if len(p) > 1 and p.lower() not in common_words]
        
        return names
    
    def _index_single_file(self, file_path: str) -> bool:
        try:
            with PILImage.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # CLIP model in sentence-transformers can encode images directly
                embedding = self.model.encode([img], show_progress_bar=False)[0]  # type: ignore
                width = img.width
                height = img.height
            
            stat = os.stat(file_path)
            file_hash = self._compute_file_hash(file_path)
            filename = os.path.basename(file_path)
            extracted_names = self._extract_names_from_filename(filename)
            
            image_id = f"{file_path}_img"
            metadata: Metadata = cast(Metadata, {
                "file_path": str(file_path),
                "file_name": str(filename),
                "file_hash": str(file_hash),
                "file_mtime": float(stat.st_mtime),
                "file_size": int(stat.st_size),
                "extracted_names": str(",".join(extracted_names)),
                "width": int(width),
                "height": int(height),
            })
            
            self.collection.add(
                ids=[image_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return False
    
    def _find_files(self, input_dir: str, extensions: List[str]) -> List[str]:
        input_path = Path(input_dir)
        files = []
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            files.extend(input_path.rglob(f'*{ext}'))
        return [str(f.resolve()) for f in files]
    
    def index_directory(self, input_dir: str, file_extensions: List[str] = ['.jpg', '.jpeg', '.png'], force: bool = False) -> Dict[str, int]:
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
        
        success_count = sum(1 for f in new_files + updated_files if self._index_single_file(f))
        
        return {
            'new': len([f for f in new_files if f in (new_files + updated_files)[:success_count]]),
            'updated': len([f for f in updated_files if f in (new_files + updated_files)[:success_count]]),
            'skipped': len(skipped_files),
            'total': len(files),
            'indexed': success_count
        }
    
    def search_by_text(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search images using text query."""
        query_lower = query.lower().strip()
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 3, 100),
            include=["metadatas", "distances"]
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        search_results = []
        for i in range(len(results['ids'][0])):
            if not results['metadatas'] or not results['distances']:
                continue
                
            metadata = cast(Metadata, results['metadatas'][0][i])
            distance = results['distances'][0][i]
            extracted_names = str(metadata.get('extracted_names', '')).lower()
            
            base_score = 1 / (1 + distance)
            final_score = base_score
            
            if query_lower in extracted_names.split(','):
                final_score = 0.95 + (base_score * 0.05)
            
            search_results.append({
                'file_path': str(metadata['file_path']),
                'file_name': str(metadata['file_name']),
                'score': final_score,
                'distance': distance,
                'extracted_names': str(metadata.get('extracted_names', '')),
            })
        
        search_results.sort(key=lambda x: x['score'], reverse=True)
        return search_results[:n_results]
    
    def search_by_image(self, image_path: str, n_results: int = 5) -> List[Dict]:
        """Search similar images using reference image."""
        try:
            with PILImage.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                query_embedding = self.model.encode([img], show_progress_bar=False)[0].tolist()  # type: ignore
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results + 1,  # +1 to potentially exclude self
                include=["metadatas", "distances"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            search_results = []
            for i in range(len(results['ids'][0])):
                if not results['metadatas'] or not results['distances']:
                    continue
                    
                metadata = cast(Metadata, results['metadatas'][0][i])
                distance = results['distances'][0][i]
                
                # Skip if it's the same image
                if str(metadata['file_path']) == os.path.abspath(image_path):
                    continue
                
                search_results.append({
                    'file_path': str(metadata['file_path']),
                    'file_name': str(metadata['file_name']),
                    'score': 1 / (1 + distance),
                    'distance': distance,
                    'extracted_names': str(metadata.get('extracted_names', '')),
                })
                
                if len(search_results) >= n_results:
                    break
            
            return search_results
            
        except Exception as e:
            print(f"Error searching by image: {e}")
            return []
    
    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Image semantic search", "model_name": self.model_name}
        )
    
    def get_stats(self) -> Dict[str, int | float]:
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return {'total_images': 0, 'total_size_mb': 0}
        
        db_path = Path(self.db_path)
        total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        
        return {
            'total_images': len(all_data['ids']),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
