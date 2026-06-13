
## Overview
A concise proof-of-concept demonstrating Retrieval-Augmented Generation (RAG) applied to web page content. It uses LangChain for orchestration, SQLite with `sqlite-vec` for vector storage, and Google Gemini for embeddings and LLM responses.

> Tip: You can use Google AI Studio to generate API Keys for free access to Google Gemini models. Visit [Google AI Studio](https://ai.google.dev/studio) to get started.

## Setup
1. Clone the repository and navigate to the project directory.
2. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. (Optional) Execute the SQLite demo script to test the dummy database creation:
   ```bash
   cd src
   sqlite_demo.py
   ```   
3. Set up environment variables for Google Gemini API credentials:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key" # On Windows use `set GOOGLE_API_KEY=your_google_api_key`
   ```
4. Initialize the database and populate it with sample data:
   ```bash
   cd src
   database.py
   ```

## Ingestion
To ingest data into the vector store, run the following command:
```bash
cd src
ingest.py
```
When prompted, enter the URL of a web page or the path to a local file when prompted. The script will extract the content, split it into chunks, generate embeddings, and store everything in the SQLite database.

## Search/Retrieval
To perform a search and retrieve relevant information, run the following command:
```bash
cd src
search.py
```
When prompted, enter a natural language query related to the ingested content. The script will convert the query into an embedding, search for similar chunks in the vector store, and use the retrieved context to generate an answer via the LLM.


## RAG Process used in this repository

This repository implements a simple Retrieval-Augmented Generation (RAG) workflow that combines a vector store (SQLite + sqlite-vec) with Google Gemini embeddings and LLMs.

- **Database:** Initialize a SQLite file with the `sqlite-vec` extension to store dense vectors (`document_vectors`) and text chunks (`document_chunks`).

- **Ingestion (indexing)**
   - Fetch input content (web page or file) and extract the HTML body or raw text.
   - Split content into overlapping chunks for robust semantic coverage (default: 1000 characters with 200-character overlap).
   - Generate embeddings for each chunk using the same embedding model used at query time.
   - Persist chunk text and the corresponding embedding vector in the DB; chunks and vectors are linked by a `document_id`.

- **Retrieval + Generation (RAG)**
   - Accept a user query and generate an embedding for the query using the same embedding model.
   - Perform a KNN search in the vector index to find the top-N nearest chunks (default: top 3).
   - Retrieve the original chunk text from `document_chunks` and assemble them as additional context.
   - Send a structured prompt (context + user question) to the LLM (Google Gemini) and return the model's answer.
   - The LLM should be instructed to answer only from the provided context and to admit when the context does not contain the answer.

- **Operational notes**
   - Ensure `GOOGLE_API_KEY` is set in your environment before running ingestion or search.
   - Verify the model's embedding dimensionality matches `VECTOR_DIMENSIONS` in `src/database.py` (update if necessary).
   - Default commands:
      ```bash
      cd src
      python database.py    # initialize DB
      python ingest.py      # ingest content and build vector index
      python search.py      # run RAG search / QA
      ```

# Conclusion
This repository demonstrates a complete RAG pipeline using Google Gemini for both embeddings and generation, with a simple SQLite vector store for retrieval. It serves as a starting point for building more complex RAG applications that can handle larger datasets, multiple documents, and more sophisticated prompting strategies.


