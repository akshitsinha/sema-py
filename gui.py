import gradio as gr
from pathlib import Path
import time
from typing import List, Tuple, Optional
from text import SemanticSearchEngine
from image import ImageSearchEngine


class SemanticSearchGUI:
    def __init__(self):
        self.text_engine: Optional[SemanticSearchEngine] = None
        self.image_engine: Optional[ImageSearchEngine] = None
        self.directory: Optional[str] = None
        
    def load_directory(self, directory: str) -> Tuple[str, dict, dict, dict, dict]:  # type: ignore
        """Load directory and initialize engines."""
        try:
            dir_path = Path(directory).expanduser().resolve()
            if not dir_path.exists():
                return ("❌ Error: Directory does not exist", {}, {}, {}, {})
            if not dir_path.is_dir():
                return ("❌ Error: Path is not a directory", {}, {}, {}, {})
            
            self.directory = str(dir_path)
            self.text_engine = SemanticSearchEngine()
            self.image_engine = ImageSearchEngine()
            
            text_files = list(dir_path.rglob("*.txt")) + list(dir_path.rglob("*.md"))
            image_files = [
                f for f in dir_path.rglob("*") 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            ]
            
            text_stats = self.text_engine.get_stats()
            image_stats = self.image_engine.get_stats()
            
            status = f"✅ **Loaded:** `{self.directory}`\n\n"
            status += f"📁 Found {len(text_files)} text files and {len(image_files)} images\n\n"
            status += f"💾 Database: {text_stats['total_chunks']} text chunks, {image_stats['total_images']} images indexed\n\n"
            
            if text_stats['total_chunks'] == 0 or image_stats['total_images'] == 0:
                status += "⚠️ **Note:** You need to index files before searching. Go to Management tab."
            
            return (
                status,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        except Exception as e:
            return (f"❌ Error: {str(e)}", {}, {}, {}, {})
    
    def index_text_files(self, progress=gr.Progress()) -> str:
        """Index text files in directory."""
        if not self.directory or not self.text_engine:
            return "❌ Please load a directory first"
        
        try:
            progress(0, desc="Finding text files...")
            dir_path = Path(self.directory)
            files = list(dir_path.rglob("*.txt")) + list(dir_path.rglob("*.md"))
            
            if not files:
                return "⚠️ No text files found"
            
            progress(0.3, desc=f"Indexing {len(files)} text files...")
            start_time = time.time()
            stats = self.text_engine.index_directory(self.directory)
            elapsed = time.time() - start_time
            
            result = f"✅ **Text Indexing Complete** ({elapsed:.2f}s)\n\n"
            result += f"📝 New: {stats['new']}\n"
            result += f"🔄 Updated: {stats['updated']}\n"
            result += f"⏭️ Skipped: {stats['skipped']}\n"
            result += f"📊 Total chunks created: {stats.get('chunks', 0)}\n"
            result += f"📂 Total files processed: {stats['total']}"
            
            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def index_images(self, progress=gr.Progress()) -> str:
        """Index images in directory."""
        if not self.directory or not self.image_engine:
            return "❌ Please load a directory first"
        
        try:
            progress(0, desc="Finding images...")
            dir_path = Path(self.directory)
            files = [
                str(f) for f in dir_path.rglob("*")
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            ]
            
            if not files:
                return "⚠️ No images found"
            
            progress(0.3, desc=f"Indexing {len(files)} images...")
            start_time = time.time()
            stats = self.image_engine.index_directory(self.directory)
            elapsed = time.time() - start_time
            
            result = f"✅ **Image Indexing Complete** ({elapsed:.2f}s)\n\n"
            result += f"🖼️ New: {stats['new']}\n"
            result += f"🔄 Updated: {stats['updated']}\n"
            result += f"⏭️ Skipped: {stats['skipped']}\n"
            result += f"📊 Total indexed: {stats['indexed']}/{stats['total']}"
            
            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def search_text(self, query: str, n_results: int) -> str:
        """Search text files."""
        if not self.text_engine:
            return "❌ Please load and index a directory first"
        if not query.strip():
            return "⚠️ Please enter a search query"
        
        try:
            # Check if database has any data
            stats = self.text_engine.get_stats()
            if stats['total_chunks'] == 0:
                return "⚠️ No indexed data found. Please go to Management tab and click 'Index Text Files' first."
            
            start_time = time.time()
            results = self.text_engine.search(query, n_results=int(n_results))
            elapsed = time.time() - start_time
            
            if not results:
                return f"🔍 No results found for: **{query}**\n\n(Searched {stats['total_chunks']} chunks in database)"
            
            output = f"🔍 **Search:** {query} ({elapsed:.2f}s, {len(results)} results)\n\n"
            output += "---\n\n"
            
            for i, result in enumerate(results, 1):
                file_name = Path(result['file_path']).name
                output += f"### {i}. {file_name}\n\n"
                output += f"**Score:** {result['score']:.3f} | **Path:** `{result['file_path']}`\n\n"
                output += f"**Lines:** {result.get('line_start', 0)}-{result.get('line_end', 0)}\n\n"
                preview = result['text'][:300].replace('\n', ' ')
                if len(result['text']) > 300:
                    preview += "..."
                output += f"> {preview}\n\n"
                output += "---\n\n"
            
            return output
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def search_images_by_text(self, query: str, n_results: int) -> Tuple[List[Tuple[str, str]], str]:
        """Search images by text query."""
        if not self.image_engine:
            return ([], "❌ Please load and index a directory first")
        if not query.strip():
            return ([], "⚠️ Please enter a search query")
        
        try:
            # Check if database has any data
            stats = self.image_engine.get_stats()
            if stats['total_images'] == 0:
                return ([], "⚠️ No indexed images found. Please go to Management tab and click 'Index Images' first.")
            
            start_time = time.time()
            results = self.image_engine.search_by_text(query, n_results=int(n_results))
            elapsed = time.time() - start_time
            
            if not results:
                return ([], f"🔍 No results found for: **{query}**\n\n(Searched {stats['total_images']} images in database)")
            
            gallery_images = []
            for result in results:
                caption = f"{result['file_name']}\nScore: {result['score']:.3f}"
                if result.get('extracted_names'):
                    caption += f"\nNames: {result['extracted_names']}"
                gallery_images.append((result['file_path'], caption))
            
            info = f"✅ Found {len(results)} images ({elapsed:.2f}s)"
            return (gallery_images, info)
        except Exception as e:
            return ([], f"❌ Error: {str(e)}")
    
    def search_images_by_reference(self, ref_image: str, n_results: int) -> Tuple[List[Tuple[str, str]], str]:
        """Search images by reference image."""
        if not self.image_engine:
            return ([], "❌ Please load and index a directory first")
        if not ref_image:
            return ([], "⚠️ Please upload a reference image")
        
        try:
            # Check if database has any data
            stats = self.image_engine.get_stats()
            if stats['total_images'] == 0:
                return ([], "⚠️ No indexed images found. Please go to Management tab and click 'Index Images' first.")
            
            start_time = time.time()
            results = self.image_engine.search_by_image(ref_image, n_results=int(n_results))
            elapsed = time.time() - start_time
            
            if not results:
                return ([], "🔍 No similar images found")
            
            gallery_images = []
            for result in results:
                caption = f"{result['file_name']}\nScore: {result['score']:.3f}"
                if result.get('extracted_names'):
                    caption += f"\nNames: {result['extracted_names']}"
                gallery_images.append((result['file_path'], caption))
            
            info = f"✅ Found {len(results)} similar images ({elapsed:.2f}s)"
            return (gallery_images, info)
        except Exception as e:
            return ([], f"❌ Error: {str(e)}")
    
    def get_stats(self) -> str:
        """Get current statistics."""
        if not self.text_engine or not self.image_engine:
            return "⚠️ No directory loaded"
        
        try:
            text_stats = self.text_engine.get_stats()
            image_stats = self.image_engine.get_stats()
            
            output = "## 📊 Database Statistics\n\n"
            output += f"**Directory:** `{self.directory}`\n\n"
            output += "### 📝 Text Files\n\n"
            output += f"- Total files: {text_stats['total_files']}\n"
            output += f"- Total chunks: {text_stats['total_chunks']}\n"
            output += f"- Database size: {text_stats['total_size_mb']} MB\n\n"
            output += "### 🖼️ Images\n\n"
            output += f"- Total images: {image_stats['total_images']}\n"
            output += f"- Database size: {image_stats['total_size_mb']} MB\n\n"
            
            return output
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_text_db(self) -> str:
        """Clear text database."""
        if not self.text_engine:
            return "⚠️ No text engine loaded"
        try:
            self.text_engine.clear()
            return "✅ Text database cleared"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_image_db(self) -> str:
        """Clear image database."""
        if not self.image_engine:
            return "⚠️ No image engine loaded"
        try:
            self.image_engine.clear()
            return "✅ Image database cleared"
        except Exception as e:
            return f"❌ Error: {str(e)}"


def create_gui():
    gui = SemanticSearchGUI()
    
    with gr.Blocks(
        title="Semantic Search",
        css="""
        .main-container {max-width: 1400px; margin: auto;}
        .header {text-align: center; padding: 2rem 0;}
        .status-box {border-left: 4px solid #2563eb; padding: 1rem; margin: 1rem 0;}
        """
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Semantic Search
            ### Local semantic search for text files and images
            """,
            elem_classes="header"
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                directory_input = gr.Textbox(
                    label="📁 Directory Path",
                    placeholder="Enter directory path (e.g., /path/to/folder or ~/Documents)",
                    lines=1
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("Load Directory", variant="primary", size="lg")
        
        status_output = gr.Markdown("", elem_classes="status-box")
        
        with gr.Tabs(visible=False) as main_tabs:
            with gr.Tab("📝 Text Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_query = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter your search query...",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        text_n_results = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Results"
                        )
                
                text_search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                text_results = gr.Markdown(label="Results")
            
            with gr.Tab("🖼️ Image Search - Text"):
                gr.Markdown("Search for images using text descriptions or names")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        img_text_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., 'john', 'person named hritik', 'beach scene'",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        img_text_n_results = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Results"
                        )
                
                img_text_search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                img_text_info = gr.Markdown("")
                img_text_gallery = gr.Gallery(
                    label="Results",
                    show_label=False,
                    columns=4,
                    object_fit="cover",
                    height="auto"
                )
            
            with gr.Tab("🖼️ Image Search - Reference"):
                gr.Markdown("Find similar images using a reference image")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        ref_image = gr.Image(
                            label="Reference Image",
                            type="filepath",
                            height=300
                        )
                    with gr.Column(scale=1):
                        ref_n_results = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Results"
                        )
                        ref_search_btn = gr.Button("🔍 Find Similar", variant="primary", size="lg")
                
                ref_info = gr.Markdown("")
                ref_gallery = gr.Gallery(
                    label="Similar Images",
                    show_label=False,
                    columns=4,
                    object_fit="cover",
                    height="auto"
                )
            
            with gr.Tab("⚙️ Management"):
                gr.Markdown("### Index and manage your search database")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📝 Text Files")
                        text_index_btn = gr.Button("Index Text Files", variant="secondary", size="lg")
                        text_index_output = gr.Markdown("")
                    
                    with gr.Column():
                        gr.Markdown("#### 🖼️ Images")
                        image_index_btn = gr.Button("Index Images", variant="secondary", size="lg")
                        image_index_output = gr.Markdown("")
                
                gr.Markdown("---")
                
                with gr.Row():
                    stats_btn = gr.Button("📊 Show Statistics", size="lg")
                    clear_text_btn = gr.Button("🗑️ Clear Text DB", variant="stop")
                    clear_image_btn = gr.Button("🗑️ Clear Image DB", variant="stop")
                
                management_output = gr.Markdown("")
        
        # Event handlers
        load_btn.click(
            fn=gui.load_directory,
            inputs=[directory_input],
            outputs=[status_output, main_tabs, main_tabs, main_tabs, main_tabs]
        )
        
        text_search_btn.click(
            fn=gui.search_text,
            inputs=[text_query, text_n_results],
            outputs=[text_results]
        )
        
        img_text_search_btn.click(
            fn=gui.search_images_by_text,
            inputs=[img_text_query, img_text_n_results],
            outputs=[img_text_gallery, img_text_info]
        )
        
        ref_search_btn.click(
            fn=gui.search_images_by_reference,
            inputs=[ref_image, ref_n_results],
            outputs=[ref_gallery, ref_info]
        )
        
        text_index_btn.click(
            fn=gui.index_text_files,
            outputs=[text_index_output]
        )
        
        image_index_btn.click(
            fn=gui.index_images,
            outputs=[image_index_output]
        )
        
        stats_btn.click(
            fn=gui.get_stats,
            outputs=[management_output]
        )
        
        clear_text_btn.click(
            fn=gui.clear_text_db,
            outputs=[management_output]
        )
        
        clear_image_btn.click(
            fn=gui.clear_image_db,
            outputs=[management_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
