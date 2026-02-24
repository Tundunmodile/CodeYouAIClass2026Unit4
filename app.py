import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Load environment variables
load_dotenv()

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    Cosine similarity = (A · B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    
    return dot_product / (norm_a * norm_b)

def search_sentences(vector_store, query, k=3):
    """
    Search for similar sentences in the vector store.

    Args:
        vector_store: The InMemoryVectorStore instance.
        query (str): The search query string.
        k (int): The number of top results to return. Defaults to 3.

    Returns:
        list: A list of tuples containing the sentence and its similarity score.
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    print("\n🔍 Search Results:")
    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"{rank}. [Score: {score:.4f}] {sentence}")

    return results

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        return

    # Initialize vector store and other components
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)

    # Add sentences to vector store
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog barked at the mailman.",
        "The bird flew over the mat.",
        "The cat sat on the windowsill.",
        "The sun is shining brightly today.",
        "The rain is pouring down outside.",
        "The children are playing in the park.",
        "The car is parked in the driveway.",
        "The book is on the table.",
        "The coffee is hot and delicious.",
        "The music is playing softly in the background.",
        "The flowers are blooming in the garden.",
        "The stars are twinkling in the night sky.",
        "The river flows gently through the valley.",
        "The mountain is covered in snow.",
        "The beach is crowded with tourists.",
        "The city lights are shining brightly at night.",
        "The food at the restaurant was amazing.",
        "The movie was thrilling and full of suspense."
    ]
    metadata = [{"created_at": datetime.now().isoformat(), "index": i} for i in range(len(sentences))]
    vector_store.add_texts(sentences, metadata=metadata)

    print(f"✅ Stored {len(sentences)} sentences in the vector store.")
    for i, sentence in enumerate(sentences, start=1):
        print(f"Sentence {i}: {sentence}")

    # Interactive semantic search loop
    print("\n=== Semantic Search ===")
    while True:
        query = input("Enter a search query (or 'quit' to exit): ").strip()

        if query.lower() in {"quit", "exit"}:
            break

        if not query:
            print("⚠️ Please enter a valid query.")
            continue

        # Perform the search and display results
        results = search_sentences(vector_store, query)
        print("\nResults:")
        for rank, (sentence, score) in enumerate(results, start=1):
            print(f"{rank}. [Score: {score:.4f}] {sentence}")

        print("\n")  # Blank line for readability

    print("Goodbye! Thanks for using the Semantic Search tool.")



if __name__ == "__main__":
    main()
