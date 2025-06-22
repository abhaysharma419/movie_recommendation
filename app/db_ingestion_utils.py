import psycopg2
import pandas as pd
import os

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="postgres",  # Service name from docker-compose.yaml
            database="airflow", # Database name from docker-compose.yaml
            user="airflow",     # User name from docker-compose.yaml
            password="airflow"  # Password from docker-compose.yaml
        )
        print("Successfully connected to PostgreSQL database.")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def create_tables():
    """Creates necessary tables in the PostgreSQL database."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                gender VARCHAR(1),
                age INTEGER,
                occupation INTEGER,
                zip_code VARCHAR(10)
            );
        """)
        # Movies table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY,
                title VARCHAR(255),
                genres VARCHAR(255)
            );
        """)
        # Ratings table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER,
                movie_id INTEGER,
                rating INTEGER,
                timestamp BIGINT,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            );
        """)
        # Merged data table (simplified, can be more detailed if needed)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS merged_movie_data (
                user_id INTEGER,
                movie_id INTEGER,
                rating INTEGER,
                timestamp TIMESTAMP,
                title VARCHAR(255),
                genres VARCHAR(255),
                gender VARCHAR(1),
                age INTEGER,
                occupation INTEGER,
                day_of_week INTEGER,
                hour_of_day INTEGER,
                PRIMARY KEY (user_id, movie_id, timestamp)
            );
        """)
        # User-Movie Matrix table (for pre-calculated matrix)
        # This will be sparse, consider a different representation for very large matrices
        # For now, we'll store user_id, movie_id, rating
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_movie_matrix (
                user_id INTEGER,
                movie_id INTEGER,
                rating INTEGER,
                PRIMARY KEY (user_id, movie_id)
            );
        """)
        # User Similarity Matrix table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_similarity_matrix (
                user_id_1 INTEGER,
                user_id_2 INTEGER,
                similarity_score FLOAT,
                PRIMARY KEY (user_id_1, user_id_2)
            );
        """)
        # User Feedback table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                feedback_type VARCHAR(10) NOT NULL, -- 'like' or 'dislike'
                feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        print("Tables created successfully (if they didn't exist).")
    except (Exception, psycopg2.Error) as error:
        print(f"Error creating tables: {error}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            cur.close()
            conn.close()

def load_processed_data_to_db(data_dir):
    """
    Loads processed data (user-movie matrix) into the PostgreSQL database.
    This function demonstrates loading the user-movie matrix as a sparse representation.
    For other dataframes (users, movies, ratings, final_merged_df),
    you would load them similarly, inserting into their respective tables.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Load original data files for users, movies, ratings to populate their tables
        # This assumes op2.py or similar functions will output these cleaned dataframes
        
        # Load users.dat
        users_filepath = os.path.join(data_dir, 'users.dat')
        users_df = pd.read_csv(users_filepath, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        print(f"Loading {users_filepath} into 'users' table...")
        for index, row in users_df.iterrows():
            cur.execute("""
                INSERT INTO users (user_id, gender, age, occupation, zip_code)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    gender = EXCLUDED.gender,
                    age = EXCLUDED.age,
                    occupation = EXCLUDED.occupation,
                    zip_code = EXCLUDED.zip_code;
            """, tuple(row))
        conn.commit()
        print("Users data loaded.")

        # Load movies.dat
        movies_filepath = os.path.join(data_dir, 'movies.dat')
        movies_df = pd.read_csv(movies_filepath, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
        print(f"Loading {movies_filepath} into 'movies' table...")
        for index, row in movies_df.iterrows():
            cur.execute("""
                INSERT INTO movies (movie_id, title, genres)
                VALUES (%s, %s, %s)
                ON CONFLICT (movie_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    genres = EXCLUDED.genres;
            """, tuple(row))
        conn.commit()
        print("Movies data loaded.")

        # Load ratings.dat
        ratings_filepath = os.path.join(data_dir, 'ratings.dat')
        ratings_df = pd.read_csv(ratings_filepath, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        print(f"Loading {ratings_filepath} into 'ratings' table...")
        for index, row in ratings_df.iterrows():
            cur.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, timestamp)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, movie_id) DO UPDATE SET
                    rating = EXCLUDED.rating,
                    timestamp = EXCLUDED.timestamp;
            """, tuple(row))
        conn.commit()
        print("Ratings data loaded.")


        # Load user_movie_matrix.csv (sparse representation)
        user_movie_matrix_filepath = os.path.join(os.path.dirname(data_dir), "processed_data", "user_movie_matrix.csv")
        if not os.path.exists(user_movie_matrix_filepath):
            raise FileNotFoundError(f"User-movie matrix CSV not found at: {user_movie_matrix_filepath}")

        user_movie_matrix_df = pd.read_csv(user_movie_matrix_filepath, index_col='UserID')
        print(f"Loading user-movie matrix from {user_movie_matrix_filepath} into 'user_movie_matrix' table...")
        
        # Iterate over the DataFrame and insert non-zero ratings
        for user_id, row in user_movie_matrix_df.iterrows():
            for movie_id_str, rating in row.items():
                movie_id = int(movie_id_str) # Convert column name (movie_id) to int
                if rating > 0: # Only insert actual ratings, not the filled zeros
                    cur.execute("""
                        INSERT INTO user_movie_matrix (user_id, movie_id, rating)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, movie_id) DO UPDATE SET
                            rating = EXCLUDED.rating;
                    """, (user_id, movie_id, int(rating)))
        conn.commit()
        print("User-Movie Matrix loaded successfully.")

        # Load user_similarity_matrix.csv
        user_similarity_matrix_filepath = os.path.join(os.path.dirname(data_dir), "processed_data", "user_similarity_matrix.csv")
        if not os.path.exists(user_similarity_matrix_filepath):
            raise FileNotFoundError(f"User similarity matrix CSV not found at: {user_similarity_matrix_filepath}")

        user_similarity_matrix_df = pd.read_csv(user_similarity_matrix_filepath, index_col='UserID')
        print(f"Loading user similarity matrix from {user_similarity_matrix_filepath} into 'user_similarity_matrix' table...")
        
        # Iterate over the DataFrame and insert similarity scores
        for user_id_1, row in user_similarity_matrix_df.iterrows():
            for user_id_2_str, score in row.items():
                user_id_2 = int(user_id_2_str) # Convert column name (user_id) to int
                # Only store one entry for each pair (e.g., if (1,2) is stored, don't store (2,1))
                if user_id_1 < user_id_2:
                    cur.execute("""
                        INSERT INTO user_similarity_matrix (user_id_1, user_id_2, similarity_score)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id_1, user_id_2) DO UPDATE SET
                            similarity_score = EXCLUDED.similarity_score;
                    """, (user_id_1, user_id_2, score))
        conn.commit()
        print("User Similarity Matrix loaded successfully.")

    except (Exception, psycopg2.Error) as error:
        print(f"Error loading data to database: {error}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            cur.close()
            conn.close()