import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api import ClientAPI

# --- Configuration ---
MODEL_NAME = "google/embeddinggemma-300m"

# ChromaDB configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "file_embeddings"

# --- Model Initialization ---
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# --- Helper Functions ---

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using the SentenceTransformer model."""
    try:
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def is_processed(client: ClientAPI, collection_name: str, file_path: str) -> bool:
    """Checks if a file has already been processed and stored in ChromaDB."""
    try:
        collection = client.get_collection(collection_name)
        result = collection.get(ids=[file_path])
        return len(result["ids"]) > 0
    except Exception:
        return False


# --- Main Logic ---
def process_directory(directory_path: str, client: ClientAPI):
    """Processes all text files in a directory, generating and storing embeddings."""
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith((".txt", ".md", ".py", ".js", ".html", ".css")): # Add other text file extensions as needed
                if is_processed(client, COLLECTION_NAME, file_path):
                    print(f"Skipping already processed file: {file_path}")
                    continue

                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if not content.strip():
                        print(f"Skipping empty file: {file_path}")
                        continue

                    embedding = get_embedding(content)
                    if embedding:
                        collection.add(
                            ids=[file_path],
                            embeddings=[embedding],
                            documents=[content]
                        )
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

def query_files(query: str, client: ClientAPI, n_results: int = 5):
    """Queries the ChromaDB for relevant files based on the user's query."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("No files have been processed yet. Please process a directory first.")
        return

    query_embedding = get_embedding(query)

    if not query_embedding:
        print("Could not generate query embedding.")
        return

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    if not results["ids"] or not results["ids"][0]:
        print("No relevant files found.")
        return

    print("\n--- Relevant Files ---")
    ids = results["ids"][0]
    distances_list = results.get('distances')
    documents_list = results.get('documents')
    
    for i, file_path in enumerate(ids):
        distance = distances_list[0][i] if distances_list else 0.0
        document = documents_list[0][i] if documents_list else ""
        
        snippet = document[:200] + "..." if len(document) > 200 else document
        snippet = snippet.replace('\n', ' ')
        
        print(f"\n{i+1}. {file_path}")
        print(f"   Similarity: {1 - distance:.2%}")
        print(f"   Snippet: {snippet}")


def main():
    """Main function to handle command-line arguments and user interaction."""
    parser = argparse.ArgumentParser(description="Generate embeddings for files and query them.")
    parser.add_argument("directory", nargs="?", default=None, help="The directory to process files from.")
    args = parser.parse_args()

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found at '{args.directory}'")
            sys.exit(1)
        process_directory(args.directory, client)
        print("\nFinished processing directory.")

    # Interactive query loop
    print("\n--- Ask a question ---")
    try:
        while True:
            query = input("> ")
            if query.lower() in ["exit", "quit"]:
                break
            query_files(query, client)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()