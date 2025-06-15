import pandas as pd
import os

# Define the base directory where you extracted the MovieLens 1M dataset
# Assuming you extracted it to 'ml-1m_dataset/ml-1m/' as per the previous step
data_dir = 'ml-1m_dataset/ml-1m/'

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
# An inner merge ensures that only ratings for movies that exist in both DataFrames are kept.
movie_ratings_df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
print("--- After merging ratings and movies DataFrames ---")
print("Head of movie_ratings_df:")
print(movie_ratings_df.head())
print(f"Shape: {movie_ratings_df.shape}\n")

# --- Step 2: Merge the result (movie_ratings_df) with users_df on 'UserID' ---
# Again, an inner merge ensures we only keep data for users present in both DataFrames.
final_merged_df = pd.merge(movie_ratings_df, users_df, on='UserID', how='inner')

print("--- Final Merged DataFrame (all three combined) ---")
print("Head of final_merged_df:")
print(final_merged_df.head())
print(f"Shape: {final_merged_df.shape}")

# Optional: Display some info about the final DataFrame
print("\nInfo about final_merged_df:")
final_merged_df.info()


# --- Data Cleaning and Preparation Steps ---

# 1. Check for and handle missing values (usually not an issue after inner merges)
print("Missing values before cleaning:")
print(final_merged_df.isnull().sum())
# If there were NaNs, you'd handle them, e.g., final_merged_df.dropna(inplace=True)


# 2. Convert Data Types
final_merged_df['Timestamp'] = pd.to_datetime(final_merged_df['Timestamp'], unit='s')
final_merged_df['UserID'] = final_merged_df['UserID'].astype(int)
final_merged_df['MovieID'] = final_merged_df['MovieID'].astype(int)
final_merged_df['Rating'] = final_merged_df['Rating'].astype(int)
print("\nData types after core conversions:")
print(final_merged_df.dtypes)


# 3. Extract Features from 'Genres' Column (One-Hot Encoding approach for content-based features)
# This creates new columns for each genre.
all_genres = final_merged_df['Genres'].str.get_dummies(sep='|')
final_merged_df = pd.concat([final_merged_df, all_genres], axis=1)
# You might drop the original 'Genres' column if you've one-hot encoded it
final_merged_df = final_merged_df.drop(columns=['Genres'], errors='ignore')
print("\nDataFrame head after one-hot encoding 'Genres':")
print(final_merged_df.head())


# 4. Convert other categorical features to 'category' dtype for memory efficiency
# Or one-hot encode them if your model requires it
final_merged_df['Gender'] = final_merged_df['Gender'].astype('category')
final_merged_df['Occupation'] = final_merged_df['Occupation'].astype('category')
# 'Age' is already grouped numerically (e.g., 1, 18, 25, etc.) so no need for explicit conversion unless using as categorical.


# 5. Check for and remove duplicate rows
print(f"\nShape before dropping duplicates: {final_merged_df.shape}")
final_merged_df.drop_duplicates(inplace=True)
print(f"Shape after dropping duplicates: {final_merged_df.shape}")


# 6. Feature Engineering from Existing Data (e.g., time-based features)
final_merged_df['DayOfWeek'] = final_merged_df['Timestamp'].dt.dayofweek
final_merged_df['HourOfDay'] = final_merged_df['Timestamp'].dt.hour
# You could also add 'Month', 'Year', etc.


# 7. Remove Irrelevant Columns
# 'Zip-code' is often not used in basic recommenders.
# 'Timestamp' itself might be dropped if only time-based features are needed, or kept for sequence modeling.
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
    index='UserID',     # Rows will be unique UserIDs
    columns='MovieID',  # Columns will be unique MovieIDs
    values='Rating'     # The values in the matrix will be the ratings
)

# Replace NaNs with 0
user_movie_matrix = user_movie_matrix.fillna(0)

print("\nUser-Item Matrix created. Head (first 5 rows and columns):")
# Displaying head of sparse matrix can be tricky, showing subset for clarity
# .iloc[:, :5] selects the first 5 columns
print(user_movie_matrix.iloc[:5, :5])
print(f"\nShape of User-Item Matrix: {user_movie_matrix.shape}")

# Verify presence of NaNs (expected for unrated movies)
print(f"Number of NaN values in the matrix: {user_movie_matrix.isnull().sum().sum()}")

