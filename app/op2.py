import pandas as pd
import os

def process_and_merge_data(data_dir='ml-1m_dataset/ml-1m/'):
    """
    Loads, cleans, merges, and prepares MovieLens 1M data.

    Args:
        data_dir (str): The directory where the MovieLens 1M dataset is extracted.
    """
    # Define the base directory where you extracted the MovieLens 1M dataset
    # For Airflow, this path needs to be absolute or relative to the working directory of the task
    # which is usually AIRFLOW_HOME. So, it's safer to rely on the passed data_dir.

    # Check if the directory exists and contains the expected files
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please ensure the MovieLens 1M dataset is downloaded and extracted.")
        print("You should have 'users.dat', 'movies.dat', and 'ratings.dat' inside this directory.")
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    required_files = ['users.dat', 'movies.dat', 'ratings.dat']
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            print(f"Error: Required file '{f}' not found in '{data_dir}'.")
            raise FileNotFoundError(f"Required file {f} not found in {data_dir}.")

    # --- 1. Load users.dat ---
    users_filepath = os.path.join(data_dir, 'users.dat')
    users_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    users_df = pd.read_csv(
        users_filepath,
        sep='::',
        engine='python', # Required for multi-character separator
        names=users_columns
    )
    print("Users DataFrame loaded. Head:")
    print(users_df.head())
    print(f"Shape: {users_df.shape}\n")


    # --- 2. Load movies.dat ---
    movies_filepath = os.path.join(data_dir, 'movies.dat')
    movies_columns = ['MovieID', 'Title', 'Genres']
    movies_df = pd.read_csv(
        movies_filepath,
        sep='::',
        engine='python', # Required for multi-character separator
        names=movies_columns,
        encoding='latin-1' # Movie titles can have special characters, latin-1 often works
    )
    print("Movies DataFrame loaded. Head:")
    print(movies_df.head())
    print(f"Shape: {movies_df.shape}\n")


    # --- 3. Load ratings.dat ---
    ratings_filepath = os.path.join(data_dir, 'ratings.dat')
    ratings_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings_df = pd.read_csv(
        ratings_filepath,
        sep='::',
        engine='python', # Required for multi-character separator
        names=ratings_columns
    )
    print("Ratings DataFrame loaded. Head:")
    print(ratings_df.head())
    print(f"Shape: {ratings_df.shape}\n")

    # --- Step 1: Merge ratings_df and movies_df on 'MovieID' ---
    movie_ratings_df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
    print("--- After merging ratings and movies DataFrames ---")
    print("Head of movie_ratings_df:")
    print(movie_ratings_df.head())
    print(f"Shape: {movie_ratings_df.shape}\n")

    # --- Step 2: Merge the result (movie_ratings_df) with users_df on 'UserID' ---
    final_merged_df = pd.merge(movie_ratings_df, users_df, on='UserID', how='inner')

    print("--- Final Merged DataFrame (all three combined) ---")
    print("Head of final_merged_df:")
    print(final_merged_df.head())
    print(f"Shape: {final_merged_df.shape}")

    # Optional: Display some info about the final DataFrame
    print("\nInfo about final_merged_df:")
    final_merged_df.info()


    # --- Data Cleaning and Preparation Steps ---

    # 1. Check for and handle missing values
    print("Missing values before cleaning:")
    print(final_merged_df.isnull().sum())


    # 2. Convert Data Types
    final_merged_df['Timestamp'] = pd.to_datetime(final_merged_df['Timestamp'], unit='s')
    final_merged_df['UserID'] = final_merged_df['UserID'].astype(int)
    final_merged_df['MovieID'] = final_merged_df['MovieID'].astype(int)
    final_merged_df['Rating'] = final_merged_df['Rating'].astype(int)
    print("\nData types after core conversions:")
    print(final_merged_df.dtypes)


    # 3. Extract Features from 'Genres' Column
    all_genres = final_merged_df['Genres'].str.get_dummies(sep='|')
    final_merged_df = pd.concat([final_merged_df, all_genres], axis=1)
    final_merged_df = final_merged_df.drop(columns=['Genres'], errors='ignore')
    print("\nDataFrame head after one-hot encoding 'Genres':")
    print(final_merged_df.head())


    # 4. Convert other categorical features to 'category' dtype
    final_merged_df['Gender'] = final_merged_df['Gender'].astype('category')
    final_merged_df['Occupation'] = final_merged_df['Occupation'].astype('category')


    # 5. Check for and remove duplicate rows
    print(f"\nShape before dropping duplicates: {final_merged_df.shape}")
    final_merged_df.drop_duplicates(inplace=True)
    print(f"Shape after dropping duplicates: {final_merged_df.shape}")


    # 6. Feature Engineering from Existing Data
    final_merged_df['DayOfWeek'] = final_merged_df['Timestamp'].dt.dayofweek
    final_merged_df['HourOfDay'] = final_merged_df['Timestamp'].dt.hour


    # 7. Remove Irrelevant Columns
    columns_to_drop = ['Zip-code']
    final_merged_df = final_merged_df.drop(columns=columns_to_drop, errors='ignore')


    print("\n--- Final Merged and Cleaned DataFrame ---")
    print(final_merged_df.head())
    print(f"Final Shape: {final_merged_df.shape}")
    print("\nFinal Data types:")
    print(final_merged_df.dtypes)

    # --- Create the User-Item Matrix ---
    print("Creating the User-Item Matrix using pivot_table...")

    user_movie_matrix = final_merged_df.pivot_table(
        index='UserID',
        columns='MovieID',
        values='Rating'
    )

    # Replace NaNs with 0
    user_movie_matrix = user_movie_matrix.fillna(0)

    print("\nUser-Item Matrix created. Head (first 5 rows and columns):")
    print(user_movie_matrix.iloc[:5, :5])
    print(f"\nShape of User-Item Matrix: {user_movie_matrix.shape}")

    # Verify presence of NaNs (expected for unrated movies)
    print(f"Number of NaN values in the matrix: {user_movie_matrix.isnull().sum().sum()}")

    # You might want to save this processed data for downstream tasks
    # For example, saving to a CSV or Parquet file in a designated output directory
    output_dir = os.path.join(os.path.dirname(data_dir), "processed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filepath = os.path.join(output_dir, "user_movie_matrix.csv")
    user_movie_matrix.to_csv(output_filepath)
    print(f"\nUser-Movie Matrix saved to: {output_filepath}")

# Remove the __main__ block as it will be called by Airflow
# if __name__ == "__main__":
#     # In a standalone run, you might want to call download_and_extract_zip first
#     # For Airflow, this will be handled by a preceding task
#     process_and_merge_data()