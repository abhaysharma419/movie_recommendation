# app/recommendation_logic.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import scipy.sparse
import numpy as np

def calculate_user_similarity(user_movie_matrix_path):
    """
    Loads the user-movie matrix and calculates pairwise cosine similarity between users.
    Assumes missing values (NaNs) in the input matrix have been filled (e.g., with 0).

    Args:
        user_movie_matrix_path (str): Path to the user-movie matrix CSV file.

    Returns:
        pandas.DataFrame: A square DataFrame representing the pairwise cosine similarity
                          between users. Index and columns are UserIDs.
    """
    if not os.path.exists(user_movie_matrix_path):
        raise FileNotFoundError(f"User-movie matrix not found at: {user_movie_matrix_path}")

    print(f"Loading user-movie sparse matrix from {user_movie_matrix_path}...")
    user_movie_matrix_sparse = scipy.sparse.load_npz(user_movie_matrix_path)
    
    # Load UserIDs (assuming they are saved in a separate CSV)
    output_dir = os.path.dirname(user_movie_matrix_path)
    user_ids_path = os.path.join(output_dir, "user_ids.csv")
    if not os.path.exists(user_ids_path):
        raise FileNotFoundError(f"UserIDs file not found at: {user_ids_path}")
    user_ids_df = pd.read_csv(user_ids_path)
    user_ids = user_ids_df['UserID'].tolist()

    print("User-movie sparse matrix loaded.")
    print(f"Shape: {user_movie_matrix_sparse.shape}")

    print("Calculating pairwise cosine similarity between users...")
    user_similarity_matrix = cosine_similarity(user_movie_matrix_sparse)
    user_similarity_matrix_df = pd.DataFrame(user_similarity_matrix,
                                             index=user_ids,
                                             columns=user_ids)
    print("User similarity matrix calculated. Head (first 5 rows and columns):")
    print(user_similarity_matrix_df.iloc[:5, :5])
    print(f"Shape: {user_similarity_matrix_df.shape}")

    # Save the similarity matrix
    output_dir = os.path.dirname(user_movie_matrix_path) # Assumes same output dir as user_movie_matrix
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists
    similarity_output_filepath = os.path.join(output_dir, "user_similarity_matrix.csv")
    user_similarity_matrix_df.to_csv(similarity_output_filepath)
    print(f"User similarity matrix saved to: {similarity_output_filepath}")

    return user_similarity_matrix_df

def generate_recommendations(user_id, user_movie_matrix_path, user_similarity_matrix_path, original_movies_path, n_recommendations=10, n_similar_users=5):
    """
    Generates movie recommendations for a target user based on user-based collaborative filtering.
    Includes explanations of why a movie was recommended.

    Args:
        user_id (int): The ID of the target user.
        user_movie_matrix_path (str): Path to the user-movie matrix CSV file.
        user_similarity_matrix_path (str): Path to the user similarity matrix CSV file.
        original_movies_path (str): Path to the original movies.dat file to get movie titles.
        n_recommendations (int): Number of top movies to recommend.
        n_similar_users (int): Number of most similar users to consider.

    Returns:
        list: A list of dictionaries, each containing recommended movie details and explanation.
    """
    print(f"Generating recommendations for UserID: {user_id}")

    # Load data
    # Load user-movie sparse matrix
    user_movie_matrix_sparse = scipy.sparse.load_npz(user_movie_matrix_path)

    # Load UserIDs and MovieIDs mappings
    output_dir = os.path.dirname(user_movie_matrix_path)
    user_ids_path = os.path.join(output_dir, "user_ids.csv")
    movie_ids_path = os.path.join(output_dir, "movie_ids.csv")

    if not os.path.exists(user_ids_path) or not os.path.exists(movie_ids_path):
        raise FileNotFoundError("UserIDs or MovieIDs mapping files not found.")

    user_ids_df = pd.read_csv(user_ids_path)
    user_ids_list = user_ids_df['UserID'].tolist()
    movie_ids_df = pd.read_csv(movie_ids_path)
    movie_ids_list = movie_ids_df['MovieID'].tolist()

    # Create mappings from ID to sparse matrix index
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids_list)}
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids_list)}
    idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}

    user_similarity_matrix = pd.read_csv(user_similarity_matrix_path, index_col='UserID')
    
    # Load original movies to get titles
    movies_df = pd.read_csv(
        original_movies_path,
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    movies_df.set_index('MovieID', inplace=True)

    # Check if target user exists
    if user_id not in user_id_to_idx:
        print(f"User {user_id} not found in the user-movie matrix.")
        return []

    target_user_idx = user_id_to_idx[user_id]
    target_user_ratings_sparse = user_movie_matrix_sparse[target_user_idx, :]
    target_user_rated_movie_indices = target_user_ratings_sparse.nonzero()[1] # Get column indices of rated movies

    # Get similar users (excluding the target user themselves)
    similar_users = user_similarity_matrix.loc[user_id].drop(user_id)
    similar_users = similar_users.sort_values(ascending=False)
    
    # Select top N most similar users
    top_similar_users = similar_users.head(n_similar_users).index.tolist()
    print(f"Top {n_similar_users} similar users for UserID {user_id}: {top_similar_users}")

    # Initialize a dictionary to store movie recommendations and their scores
    movie_scores = {}
    movie_explanations = {}

    for similar_uid in top_similar_users:
        if similar_uid not in user_id_to_idx:
            continue

        similar_user_idx = user_id_to_idx[similar_uid]
        similar_user_ratings_sparse = user_movie_matrix_sparse[similar_user_idx, :]

        # Find movies rated by the similar user but not by the target user
        # Get movie indices rated by similar user
        similar_user_rated_movie_indices = similar_user_ratings_sparse.nonzero()[1]

        # Convert target user's rated movie indices to a set for faster lookup
        target_user_rated_movie_indices_set = set(target_user_rated_movie_indices)

        candidate_movie_indices = []
        for movie_idx in similar_user_rated_movie_indices:
            if movie_idx not in target_user_rated_movie_indices_set:
                candidate_movie_indices.append(movie_idx)

        for movie_idx in candidate_movie_indices:
            rating_by_similar = similar_user_ratings_sparse[0, movie_idx] # Access value from sparse matrix
            movie_id = idx_to_movie_id[movie_idx]
            
            if movie_id not in movie_scores:
                movie_scores[movie_id] = {'total_rating': 0, 'count': 0}
                movie_explanations[movie_id] = []
            
            movie_scores[movie_id]['total_rating'] += rating_by_similar
            movie_scores[movie_id]['count'] += 1
            movie_explanations[movie_id].append({'user_id': similar_uid, 'rating': int(rating_by_similar)})

    # Calculate average score for each candidate movie
    recommended_movies = []
    for movie_id, data in movie_scores.items():
        if data['count'] > 0:
            avg_rating = data['total_rating'] / data['count']
            recommended_movies.append({
                'MovieID': movie_id,
                'Average_Score': avg_rating,
                'Explanation': movie_explanations[movie_id]
            })

    # Sort by average score (descending) and take top N
    recommended_movies.sort(key=lambda x: x['Average_Score'], reverse=True)
    final_recommendations = recommended_movies[:n_recommendations]

    # Add movie titles to recommendations and format explanation
    output_recommendations = []
    for rec in final_recommendations:
        movie_title = movies_df.loc[rec['MovieID']]['Title'] if rec['MovieID'] in movies_df.index else "Unknown Title"
        explanation_str_parts = []
        for exp in rec['Explanation']:
            explanation_str_parts.append(f"User {exp['user_id']} rated it {exp['rating']}/5")
        
        explanation_text = f"Because similar users ({', '.join(explanation_str_parts)})"

        output_recommendations.append({
            'MovieID': int(rec['MovieID']),
            'Title': movie_title,
            'Recommended_Score': round(rec['Average_Score'], 2),
            'Explanation': explanation_text,
            'Influencing_Users': rec['Explanation'] # Keep raw data for API
        })
    
    print(f"Generated {len(output_recommendations)} recommendations.")
    return output_recommendations