import os
from sentence_transformers import SentenceTransformer

def download_model():
    """Download the embedding model during Docker build."""
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"Downloading model: {model_name}...")
    # This will download the model to the default cache directory (or HF_HOME)
    SentenceTransformer(model_name)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
