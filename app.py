import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
from datetime import datetime
from langchain_core.documents import Document

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

def search_sentences(vector_store, query, k=3, category=None):
    """
    Search for similar sentences in the vector store with optional category filtering.

    Args:
        vector_store: The InMemoryVectorStore instance.
        query (str): The search query string.
        k (int): The number of top results to return. Defaults to 3.
        category (str, optional): The category to filter results by. Defaults to None.

    Returns:
        list: A list of tuples containing the sentence and its similarity score.
    """
    if category:
        results = vector_store.similarity_search_with_score(
            query, k=k, filter={"category": category}
        )
    else:
        results = vector_store.similarity_search_with_score(query, k=k)

    print("\n🔍 Search Results:")
    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"{rank}. [Score: {score:.4f}] {sentence}")

    return results

def hybrid_search(vector_store, query, k=3, category=None, similarity_threshold=0.5):
    """
    Perform a hybrid search combining vector similarity and keyword matching.

    Args:
        vector_store: The InMemoryVectorStore instance.
        query (str): The search query string.
        k (int): The number of top results to return. Defaults to 3.
        category (str, optional): The category to filter results by. Defaults to None.
        similarity_threshold (float): The minimum similarity score to include a result. Defaults to 0.5.

    Returns:
        list: A list of tuples containing the sentence and its combined score.
    """
    # Vector similarity search
    if category:
        vector_results = vector_store.similarity_search_with_score(
            query, k=k, filter={"category": category}
        )
    else:
        vector_results = vector_store.similarity_search_with_score(query, k=k)

    # Filter vector results by similarity threshold
    vector_results = [(sentence, score) for sentence, score in vector_results if score >= similarity_threshold]

    # Keyword matching
    keyword_results = []
    for sentence in vector_store.texts:
        if query.lower() in sentence.lower():
            keyword_results.append((sentence, 1.0))  # Assign a perfect score for keyword matches

    # Combine results
    combined_results = {sentence: score for sentence, score in vector_results}
    for sentence, score in keyword_results:
        if sentence in combined_results:
            combined_results[sentence] += score  # Boost score for keyword matches
        else:
            combined_results[sentence] = score

    # Sort by combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

    print("\n🔍 Hybrid Search Results:")
    for rank, (sentence, score) in enumerate(sorted_results[:k], start=1):
        print(f"{rank}. [Score: {score:.4f}] {sentence}")

    return sorted_results[:k]

def load_document(vector_store, file_path):
    """
    Load a document into the vector store.

    Args:
        vector_store: The InMemoryVectorStore instance.
        file_path (str): The path to the file to load.

    Returns:
        str: The document ID.
    """
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            content = file.read()

        # Create a Document object
        document = Document(
            page_content=content,
            metadata={
                'fileName': os.path.basename(file_path),
                'createdAt': datetime.now().isoformat()
            }
        )

        # Add the document to the vector store
        vector_store.add_documents([document])

        # Print success message
        print(f"✅ Successfully added document '{os.path.basename(file_path)}' with content length {len(content)}.")

        # Return the document ID (assuming the vector store generates one)
        return document.metadata['fileName']

    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

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

    # Display header
    print("=== Loading Documents into Vector Database ===")

    # Load the document
    document_id = load_document(vector_store, "/Users/tundun/CodeYouAIClass2026Unit4/HealthInsuranceBrochure.md")

    if document_id:
        print(f"Document '{document_id}' loaded successfully into the vector store.")

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
