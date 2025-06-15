from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

# Add the 'app' directory to the Python path so Airflow can find op1.py and op2.py
# This is crucial when running in a Dockerized Airflow environment
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
sys.path.insert(0, os.path.join(AIRFLOW_HOME, 'app'))

from op1 import download_and_extract_zip
from op2 import process_and_merge_data

with DAG(
    dag_id='movie_recommendation_etl',
    start_date=datetime(2023, 1, 1),
    schedule=None,  # Run manually or on a specific schedule (e.g., '@daily')
    catchup=False,
    tags=['movie_recommendation', 'etl'],
    doc_md="""
    ### Movie Recommendation ETL DAG
    This DAG downloads the MovieLens 1M dataset, processes and merges the data.
    """
) as dag:
    # Task 1: Download and Extract MovieLens Dataset
    download_task = PythonOperator(
        task_id='download_and_extract_movielens',
        python_callable=download_and_extract_zip,
        op_kwargs={
            'url': "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            'download_dir': f"{AIRFLOW_HOME}/data/temp_downloads",
            'extract_dir': f"{AIRFLOW_HOME}/data/ml-1m_dataset"
        },
    )

    # Task 2: Process and Merge Data
    process_data_task = PythonOperator(
        task_id='process_and_merge_data',
        python_callable=process_and_merge_data,
        op_kwargs={
            # Pass the full path where the data is actually extracted
            'data_dir': f"{AIRFLOW_HOME}/data/ml-1m_dataset/ml-1m/" # <--- Add this line!
        },
        # op_kwargs can be used to pass arguments if needed, 
        # but op2.py currently assumes a fixed path relative to itself.
        # For a robust solution, consider passing the data_dir as an argument here.
    )

    # Define the task dependencies
    download_task >> process_data_task