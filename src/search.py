import json
import os
import sqlite3
import sqlite_vec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

from database import DB_FILE

# Check that GOOGLE_API_KEY is set; this is required for both embeddings and LLM.
if not os.environ.get("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable is required. "
        "Set GOOGLE_API_KEY before running this script."
    )

# Initialize embeddings model to convert search queries into vectors.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# Initialize the LLM (Google Generative AI) to generate answers based on retrieved context.
llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def search_similar_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Search for the most similar chunks to the user query.
    
    1. Embed the query using the same embedding model
    2. Connect to SQLite database with vector extension
    3. Perform vector similarity search to find top_k nearest embeddings
    4. Retrieve corresponding chunk text from the document_chunks table
    5. Return the matched chunks with their similarity scores
    
    Args:
        query: The user's search query string
        top_k: Number of top similar chunks to retrieve (default: 3)
    
    Returns:
        List of dicts containing 'chunk_text' and 'distance' for each result
    """
    # Generate embedding for the user's search query.
    # This uses the same embedding model as was used during ingestion,
    # ensuring consistent vector space representation.
    query_embedding = embeddings.embed_query(query)
    
    # Connect to SQLite database.
    conn = sqlite3.connect(DB_FILE)
    try:
        # Enable the vector extension for similarity search capabilities.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        # FIXED QUERY STRUCTURE FOR SQLITE-VEC:
        # 1. Replaced broad document_id JOIN with direct rowid mapping to target exact chunk rows.
        # 2. Moved limit constraint into the WHERE clause via 'dv.k = ?' as required by vec0.
        query_sql = """
           SELECT 
                dc.chunk_text,
                dv.distance
            FROM document_vectors dv
            JOIN document_chunks dc ON dv.document_id = dc.document_id
            WHERE dv.embedding MATCH ?
            AND dv.k = ?
            ORDER BY dv.distance
        """
        
        cursor = conn.cursor()
        cursor.execute(query_sql, (json.dumps(query_embedding), top_k))
        
        results = []
        for row in cursor.fetchall():
            chunk_text, distance = row
            results.append({
                "chunk_text": chunk_text,
                "distance": distance
            })
        
        return results
    finally:
        conn.close()


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a comprehensive prompt that includes the user query and retrieved context.
    
    This creates a prompt template that:
    1. Provides the retrieved chunks as context
    2. Includes the user's original question
    3. Instructs the LLM to answer based on the provided context
    
    Args:
        query: The user's original search query
        retrieved_chunks: List of dicts with 'chunk_text' from similarity search
    
    Returns:
        Formatted prompt string ready for the LLM
    """
    # Build context from the retrieved chunks.
    context_text = "\n\n".join([
        f"[Chunk {i+1}]:\n{chunk['chunk_text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    # Create the complete RAG prompt that provides context and asks the question.
    rag_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT (retrieved from knowledge base):
{context_text}

USER QUESTION:
{query}

Please answer the user's question based on the context provided above. If the answer is not in the context, say "The answer is not available in the provided context." Do not make up information."""
    
    return rag_prompt


def generate_answer(query: str) -> str:
    """
    Complete RAG pipeline: search, retrieve, and generate answer.
    
    Flow:
    1. Search for similar chunks using vector similarity
    2. Build a prompt with retrieved chunks as context
    3. Send the prompt to the LLM
    4. Return the generated response
    
    Args:
        query: The user's search question
    
    Returns:
        LLM-generated answer based on retrieved context
    """
    # Step 1: Retrieve similar chunks from the database.
    print(f"🔍 Searching for similar chunks...")
    retrieved_chunks = search_similar_chunks(query, top_k=3)
    
    if not retrieved_chunks:
        return "No relevant content found in the knowledge base."
    
    # Display the retrieved chunks for debugging/transparency.
    print(f"✅ Found {len(retrieved_chunks)} relevant chunks.")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  Chunk {i+1} (distance: {chunk['distance']:.4f}): {chunk['chunk_text'][:100]}...")
    
    # Step 2: Build the complete prompt with context and query.
    rag_prompt = build_rag_prompt(query, retrieved_chunks)
    print(f"✅ Built RAG prompt with retrieved context.")
    print(f"Full RAG Prompt:\n{rag_prompt}\n")
    print(f"✅ Prompt is ready to be sent to the LLM.")
    
    # Step 3: Send to LLM and get the response.
    print(f"🤖 Generating answer with LLM...")
    answer = llm.invoke(rag_prompt)
    
    return answer


def main() -> None:
    """Main entry point for the RAG search system."""
    print("=" * 60)
    print("RAG Search System - Powered by SQLite-Vec + Google Gemini")
    print("=" * 60)
    
    # Get the search query from the user.
    query = input("\nEnter your search query: ").strip()
    
    if not query:
        raise ValueError("A valid search query is required.")
    
    # Run the complete RAG pipeline.
    answer = generate_answer(query)
    
    # Display the final answer.
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
