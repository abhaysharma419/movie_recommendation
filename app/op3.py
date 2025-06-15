import pandas as pd
import os

# --- IMPORTANT: Ensure these files are in 'ml-1m_dataset/ml-1m/' relative to your script ---
# If you haven't run the download and extraction step, please do so first.
# For example:
# from your_download_script import download_and_extract_zip
# download_and_extract_zip("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "temp_downloads", "ml-1m_dataset")

# Define the base directory where you extracted the MovieLens 1M dataset
data_dir = 'ml-1m_dataset/ml-1m/'

# Check if the directory exists and contains the expected files
if not os.path.exists(data_dir):
    print(f"Error: Data directory '{data_dir}' not found.")
    print("Please ensure you have downloaded and extracted the MovieLens 1M dataset.")
    print("You should have 'users.dat', 'movies.dat', and 'ratings.dat' inside 'ml-1m_dataset/ml-1m/'.")
else:
    required_files = ['users.dat', 'movies.dat', 'ratings.dat']
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            print(f"Error: Required file '{f}' not found in '{data_dir}'.")
            print("Please check your extraction process.")
            exit() # Exit if files are not found

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

