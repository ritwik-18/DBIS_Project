"""
Embedding Generator

Responsible for converting document text into vector embeddings 
using pre-trained models.
"""

def generate_embedding(text: str) -> list[float]:
    """
    Takes a string of text and returns a vector embedding.
    
    TODO: Integrate sentence-transformers (e.g., 'all-MiniLM-L6-v2') 
    to replace this dummy data.
    """
    # Placeholder for the 384-dimensional vector
    dummy_vector = [0.0] * 384 
    return dummy_vector

if __name__ == "__main__":
    sample_text = "Optimizing hybrid search in vector databases."
    vector = generate_embedding(sample_text)
    print(f"Generated vector of length: {len(vector)}")