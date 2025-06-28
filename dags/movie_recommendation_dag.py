# This DAG downloads the MovieLens 1M dataset, processes it, calculates user similarity,
# and generates movie recommendations using user-based collaborative filtering.
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

# Add the 'app' directory to the Python path so Airflow can find op1.py, op2.py, and recommendation_logic.py
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
sys.path.insert(0, os.path.join(AIRFLOW_HOME, 'app'))

from op1 import download_and_extract_zip
from op2 import process_and_merge_data
from recommendation_logic import calculate_user_similarity, generate_recommendations

with DAG(
    dag_id='movie_recommendation_etl',
    start_date=datetime(2025, 1, 1),
    schedule=None,  # Run manually or on a specific schedule (e.g., '@daily')
    catchup=False,
    tags=['movie_recommendation', 'etl'],
    doc_md="""
    ### Movie Recommendation ETL DAG
    This DAG downloads the MovieLens 1M dataset, processes and merges the data,
    calculates user similarity, and generates batch movie recommendations.
    """
) as dag:
    # Define common paths
    DOWNLOAD_DIR = f"{AIRFLOW_HOME}/data/temp_downloads"
    EXTRACT_DIR = f"{AIRFLOW_HOME}/data/ml-1m_dataset"
    PROCESSED_DATA_DIR = f"{EXTRACT_DIR}/ml-1m/processed_data" # Directory where user_movie_matrix.csv is saved by op2.py
    MOVIELENS_DATA_DIR = f"{EXTRACT_DIR}/ml-1m/" # Original extracted data directory

    # Task 1: Download and Extract MovieLens Dataset
    download_task = PythonOperator(
        task_id='download_and_extract_movielens',
        python_callable=download_and_extract_zip,
        op_kwargs={
            'url': "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            'download_dir': DOWNLOAD_DIR,
            'extract_dir': EXTRACT_DIR
        },
    )

    # Task 2: Process and Merge Data
    process_data_task = PythonOperator(
        task_id='process_and_merge_data',
        python_callable=process_and_merge_data,
        op_kwargs={
            'data_dir': MOVIELENS_DATA_DIR # Pass the full path where the data is extracted
        },
    )

    # Task 3: Calculate User Similarity
    # This task will take the user-movie matrix from process_data_task's output
    # and save the user_similarity_matrix.csv in the same directory.
    calculate_similarity_task = PythonOperator(
        task_id='calculate_user_similarity',
        python_callable=calculate_user_similarity,
        op_kwargs={
            'user_movie_matrix_path': os.path.join(PROCESSED_DATA_DIR, "user_movie_matrix.npz")
        },
    )

    # Task 4: Generate Batch Recommendations (example for a single user, can be expanded)
    # In a real scenario, you might iterate over all users or a subset.
    # For now, let's generate for a sample user.
    # This task would save the pre-calculated recommendations to a file.
    generate_batch_recommendations_task = PythonOperator(
        task_id='generate_batch_recommendations',
        python_callable=generate_recommendations,
        op_kwargs={
            'user_id': 1, # Example user ID
            'user_movie_matrix_path': os.path.join(PROCESSED_DATA_DIR, "user_movie_matrix.npz"),
            'user_similarity_matrix_path': os.path.join(PROCESSED_DATA_DIR, "user_similarity_matrix.csv"),
            'original_movies_path': os.path.join(MOVIELENS_DATA_DIR, "movies.dat"), # Path to original movies.dat
            'n_recommendations': 10,
            'n_similar_users': 5
        },
    )

    # Define the task dependencies
    download_task >> process_data_task >> calculate_similarity_task >> generate_batch_recommendations_task