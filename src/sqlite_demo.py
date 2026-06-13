import sqlite3

# 1. Establish a connection to the database file
# If 'company.db' does not exist, SQLite will create it automatically.
with sqlite3.connect("../data/company.db") as connection:
    
    # 2. Create a cursor object to execute SQL statements
    cursor = connection.cursor()
    
    # 3. Define the SQL query using 'IF NOT EXISTS' to avoid duplication errors
    create_table_query = """
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT,
        salary REAL,
        hire_date TEXT DEFAULT CURRENT_DATE
    );
    """
    
    # 4. Execute the query
    cursor.execute(create_table_query)
    
    # 5. Commit changes to save the table structure permanently
    connection.commit()

print("Database and 'employees' table successfully verified/created!")
