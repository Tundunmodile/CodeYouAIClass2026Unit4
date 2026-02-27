import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Verify GitHub token is loaded
if os.getenv("GITHUB_TOKEN"):
    print(f"✅ GitHub Token loaded successfully")
else:
    print("⚠️ GitHub Token not found in environment variables. Set GITHUB_TOKEN as an environment variable.")

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
        if "maximum context length" in str(e).lower() or "token" in str(e).lower():
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ An unexpected error occurred: {str(e)}")

def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Load document chunks into the vector store.

    Args:
        vector_store: The InMemoryVectorStore instance.
        file_path (str): The path to the file.
        chunks (list): A list of LangChain Document objects.

    Returns:
        int: The total number of chunks stored.
    """
    try:
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            # Ensure the chunk is a Document object and update metadata
            if isinstance(chunk, Document):
                chunk.metadata.update({
                    'fileName': f"{os.path.basename(file_path)} (Chunk {index}/{total_chunks})",
                    'createdAt': datetime.now().isoformat(),
                    'chunkIndex': index
                })

                # Add the chunk to the vector store
                vector_store.add_documents([chunk])

                # Print progress
                print(f"✅ Processed chunk {index}/{total_chunks} for file '{os.path.basename(file_path)}'.")
            else:
                print(f"⚠️ Skipping invalid chunk at index {index}: Not a Document object.")

        return total_chunks

    except Exception as e:
        print(f"❌ An error occurred while processing chunks: {str(e)}")
        return 0

def load_with_paragraph_chunking(vector_store, file_path):
    """
    Load the Employee Handbook with paragraph-based chunking.

    Args:
        vector_store: The InMemoryVectorStore instance.
        file_path (str): The path to the Employee Handbook file.
    """
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            text = file.read()

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )

        # Create document chunks
        chunks = text_splitter.create_documents([text])

        # Pass chunks to the vector store
        total_chunks = load_document_with_chunks(vector_store, file_path, chunks)

        # Calculate statistics
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        smallest_chunk = min(chunk_sizes) if chunk_sizes else 0
        largest_chunk = max(chunk_sizes) if chunk_sizes else 0
        chunks_starting_with_newline = sum(1 for chunk in chunks if chunk.page_content.startswith("\n"))

        # Print statistics
        print("\n=== Paragraph-Based Chunking Statistics ===")
        print(f"Total chunks created: {total_chunks}")
        print(f"Smallest chunk size: {smallest_chunk}")
        print(f"Largest chunk size: {largest_chunk}")
        print(f"Chunks starting with a newline: {chunks_starting_with_newline}")

    except Exception as e:
        print(f"\u274c An error occurred while processing the document: {str(e)}")

# def load_with_markdown_chunking(vector_store, file_path):
#     """
#     Load the Employee Handbook with markdown-based chunking.

#     Args:
#         vector_store: The InMemoryVectorStore instance.
#         file_path (str): The path to the Employee Handbook file.
#     """
#     try:
#         # Read the file content
#         with open(file_path, 'r') as file:
#             text = file.read()

#         # Initialize the MarkdownHeaderTextSplitter
#         markdown_splitter = MarkdownHeaderTextSplitter(
#             headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
#         )

#         # Split the document by markdown headers
#         header_chunks = markdown_splitter.split_text(text)

#         # Initialize the RecursiveCharacterTextSplitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=5000,
#             chunk_overlap=200
#         )

#         # Further split the header chunks into smaller chunks
#         chunks = text_splitter.create_documents(header_chunks)

#         # Pass chunks to the vector store
#         total_chunks = load_document_with_chunks(vector_store, file_path, chunks)

#         # Print statistics
#         chunk_sizes = [len(chunk.page_content) for chunk in chunks]
#         smallest_chunk = min(chunk_sizes) if chunk_sizes else 0
#         largest_chunk = max(chunk_sizes) if chunk_sizes else 0
#         chunks_starting_with_newline = sum(1 for chunk in chunks if chunk.page_content.startswith("\n"))

#         print("\n=== Markdown-Based Chunking Statistics ===")
#         print(f"Total chunks created: {total_chunks}")
#         print(f"Smallest chunk size: {smallest_chunk}")
#         print(f"Largest chunk size: {largest_chunk}")
#         print(f"Chunks starting with a newline: {chunks_starting_with_newline}")

#     except Exception as e:
#         print(f"❌ An error occurred while processing the document: {str(e)}")

# def load_with_fixed_size_chunking(vector_store, file_path):
#     """
#     Load the Employee Handbook with fixed-size chunking.

#     Args:
#         vector_store: The InMemoryVectorStore instance.
#         file_path (str): The path to the Employee Handbook file.
#     """
#     try:
#         # Read the file content
#         with open(file_path, 'r') as file:
#             text = file.read()

#         # Initialize the CharacterTextSplitter
#         text_splitter = CharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=0,
#             separator=" "
#         )

#         # Create document chunks
#         chunks = text_splitter.create_documents([text])

#         # Pass chunks to the vector store
#         total_chunks = load_document_with_chunks(vector_store, file_path, chunks)

#         # Calculate statistics
#         chunk_sizes = [len(chunk.page_content) for chunk in chunks]
#         average_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

#         # Print statistics
#         print("\n=== Fixed-Size Chunking Statistics ===")
#         print(f"Total chunks created: {total_chunks}")
#         print(f"Average chunk size: {average_chunk_size:.2f}")

#     except Exception as e:
#         print(f"❌ An error occurred while processing the document: {str(e)}")

# Function to create a search tool
def create_search_tool(vector_store):
    """
    Creates a LangChain Tool for searching documents in the vector store.

    Args:
        vector_store: The vector store to perform similarity searches on.

    Returns:
        A LangChain Tool for searching documents.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Searches the company document repository for relevant information based on the given query.
        Use this to find information about company policies, benefits, and procedures.

        Args:
            query: The query string to search for.

        Returns:
            A formatted string of the top 3 results with their content and scores.
        """
        results = vector_store.similarity_search_with_score(query, k=3)
        formatted_results = "\n\n".join(
            [
                f"Result {i + 1} (Score: {score:.4f}): {content}"
                for i, (content, score) in enumerate(results)
            ]
        )
        return formatted_results

    return search_documents

def create_react_agent_with_search(vector_store):
    """
    Creates a ReAct agent using LangChain's ReAct pattern and a search tool.

    Args:
        vector_store: The vector store to perform similarity searches on.

    Returns:
        An AgentExecutor instance for the ReAct agent.
    """
    try:
        # Create the search tool
        search_tool = create_search_tool(vector_store)

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers employee questions about company policies, benefits, and procedures.

When responding:
- Always search the internal knowledge base before answering.
- Base your answer only on retrieved documents.
- Cite the document titles or chunk identifiers used.
- If no relevant information is found, say you could not locate the answer.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create the ReAct agent
        react_agent = create_agent(
            tools=[search_tool],
            model="gpt-4o",  # Updated to use 'llm' instead of 'prompt'
            debug=True
        )

        # Wrap the agent in an executor
        return react_agent

    except Exception as e:
        print(f"❌ An error occurred while creating the ReAct agent: {str(e)}")
        raise

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub Token
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

    # Initialize ChatOpenAI instance
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )

    # Display header
    print("=== Loading Documents into Vector Database ===")

    # # Load the Employee Handbook document with fixed-size chunking
    # load_with_fixed_size_chunking(vector_store, "/Users/tundun/CodeYouAIClass2026Unit4/EmployeeHandbook.md")

    # Load the Employee Handbook document with paragraph-based chunking
    load_with_paragraph_chunking(vector_store, "/Users/tundun/CodeYouAIClass2026Unit4/EmployeeHandbook.md")

    # # Load the Employee Handbook document with markdown-based chunking
    # load_with_markdown_chunking(vector_store, "/Users/tundun/CodeYouAIClass2026Unit4/EmployeeHandbook.md")

    # Initialize the ReAct agent
    agent_executor = create_react_agent_with_search(vector_store)
    print("ReAct Agent initialized successfully.")

    # Chat interface replacing semantic search
    print("\n=== Welcome to the Chat Interface ===")
    print("I am your assistant. Ask me anything, and I will use my tools to help you. Type 'quit' or 'exit' to end the chat.")

    chat_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye! Thanks for chatting with me.")
            break

        if not user_input:
            print("⚠️ Please enter a valid message.")
            continue

        try:
            # Call the agent executor with user input and chat history
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # Extract and display the agent's response
            agent_response = result["output"]
            print(f"Agent: {agent_response}")

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")

    print("Goodbye! Thanks for using the Semantic Search tool.")

    # Initialize the ReAct agent
    # react_agent_executor = create_react_agent_with_search(vector_store)
    # print("ReAct Agent initialized successfully.")



if __name__ == "__main__":
    main()
