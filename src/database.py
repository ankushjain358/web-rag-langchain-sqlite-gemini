import os
import json
import sqlite3
import sqlite_vec

# Define configuration constants
DB_FILE = "../data/rag-app.db"
VECTOR_DIMENSIONS = 3072  # Matches typical models like all-MiniLM-L6-v2

def init_database():
    """
    Initializes the SQLite database, loads the vector extension, 
    and idempotently creates the required tables and indexes.
    """
    # 1. Connect to the SQLite database file
    # This automatically creates the file 'rag-app.db' if it does not exist.
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # 2. Enable external extensions and load the vector engine
        # This security toggle must be run on every new database connection.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        # 3. Create the Virtual Table for Vector Embeddings
        # 'IF NOT EXISTS' makes this operation idempotent (safe to run repeatedly).
        # We specify float32 and the dimensions to allocate the structural blocks safely.
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_vectors USING vec0(
                document_id INTEGER PRIMARY KEY,
                embedding float32[{VECTOR_DIMENSIONS}]
            );
        """)
        
        # 4. Create the Relational Data Table for Chunks
        # This standard B-Tree table stores the actual text chunks.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL
            );
        """)
        
        # 5. Create a Standard B-Tree Index on document_id
        # Crucial for relational performance when joining text chunks back to vector search hits.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
            ON document_chunks(document_id);
        """)
        
        # Commit the schema transactions safely to disk
        conn.commit()
        print(f"✅ Successfully initialized '{DB_FILE}' with all tables and indexes.")
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database operations failed: {e}")
        conn.rollback()
    finally:
        # Always close the connection when setup script tasks complete
        conn.close()

def insert_sample_data():
    """
    Inserts dummy data to demonstrate the structural pipeline safely.
    Uses structural integrity checks to remain idempotent.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    
    # ID check to see if our example rows already exist
    sample_id = 999
    cursor = conn.execute("SELECT 1 FROM document_chunks WHERE document_id = ? LIMIT 1", (sample_id,))
    
    if cursor.fetchone() is None:
        # Generate a dummy vector array padded out to the correct size
        dummy_vector = [0.15, -0.23, 0.88] + [0.0] * (VECTOR_DIMENSIONS - 3)
        sample_text = "SQLite-vec combined with text chunks makes a highly efficient embedded RAG workflow."
        
        try:
            # Insert vector data into the virtual table
            # We pass the float array serialized cleanly into a JSON string format
            conn.execute(
                "INSERT INTO document_vectors(document_id, embedding) VALUES (?, ?)",
                (sample_id, json.dumps(dummy_vector))
            )
            
            # Insert corresponding readable metadata inside the text chunk relational table
            conn.execute(
                "INSERT INTO document_chunks(document_id, chunk_text) VALUES (?, ?)",
                (sample_id, sample_text)
            )
            
            conn.commit()
            print(f"🚀 Inserted sample document records (ID: {sample_id}).")
        except Exception as e:
            print(f"❌ Insertion failed: {e}")
            conn.rollback()
    else:
        print(f"ℹ️ Sample records (ID: {sample_id}) already exist. Skipping insertion step.")
        
    conn.close()

if __name__ == "__main__":
    # Execute database generation pipeline
    init_database()
    # insert_sample_data()
