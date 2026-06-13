import json
import os
import sqlite3
import uuid
import bs4
import requests
import sqlite_vec
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from database import DB_FILE, VECTOR_DIMENSIONS

if not os.environ.get("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable is required. "
        "Set GOOGLE_API_KEY before running this script."
    )

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# HTML body-only parser to avoid page chrome, scripts, and unrelated page elements.
bs4_strainer = bs4.SoupStrainer("body")


def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    """Load a webpage and return the HTML body text as a LangChain Document."""
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    body = soup.body
    page_content = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
    return [Document(page_content=page_content, metadata={"source": url})]


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks for better semantic retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings for a list of text chunks."""
    return embeddings.embed_documents(texts)


def _generate_document_id() -> int:
    """Generate a unique integer ID for each chunk/document pair.

    Returns a positive 63-bit integer to safely fit into SQLite INTEGER
    (signed 64-bit). We mask the UUID int to ensure it does not exceed
    SQLite's integer range.
    """
    # Ensure the generated integer fits in SQLite's signed 64-bit INTEGER range.
    # We mask the UUID integer to 63 bits (positive range 0 .. 2**63-1).
    # This avoids the `Python int too large to convert to SQLite INTEGER` error.
    return uuid.uuid4().int & ((1 << 63) - 1)


def persist_chunks_and_embeddings(chunks: list[Document], embeddings_list: list[list[float]]) -> None:
    """Persist text chunks and their embeddings into the SQLite database."""
    if len(chunks) != len(embeddings_list):
        raise ValueError("Number of chunks and embeddings must match.")

    conn = sqlite3.connect(DB_FILE)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        with conn:
            cursor = conn.cursor()
            for chunk, vector in zip(chunks, embeddings_list):
                if len(vector) != VECTOR_DIMENSIONS:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} does not match DB vector dimension {VECTOR_DIMENSIONS}. "
                        "Update database.py VECTOR_DIMENSIONS to match your model."
                    )

                document_id = _generate_document_id()
                cursor.execute(
                    "INSERT INTO document_chunks(document_id, chunk_text) VALUES (?, ?)",
                    (document_id, chunk.page_content),
                )
                cursor.execute(
                    "INSERT INTO document_vectors(document_id, embedding) VALUES (?, ?)",
                    (document_id, json.dumps(vector)),
                )

        print(f"✅ Inserted {len(chunks)} chunks and embeddings into '{DB_FILE}'.")
    finally:
        conn.close()


def main() -> None:
    """Main pipeline for ingestion, embedding, and database insertion."""

    url = input("Enter the URL to ingest: ").strip()
    if not url:
        raise ValueError("A valid URL is required to ingest content.")

    #  1. Create the LangChain document from the text content
    docs = load_web_page(url, bs_kwargs={"parse_only": bs4_strainer})

    if not docs:
        raise RuntimeError("No documents were loaded from the web page.")

    print(f"Loaded document from {url} ({len(docs[0].page_content)} characters)")

    # 2. Create smaller chunks from that document
    chunks = chunk_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")

    # 3. Generate embeddings for each chunk
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings_list = generate_embeddings(chunk_texts)
    print(f"Generated embeddings for {len(embeddings_list)} chunks.")

    # 4. Finally, store each chunk and respective embedding into database
    persist_chunks_and_embeddings(chunks, embeddings_list)


if __name__ == "__main__":
    main()
