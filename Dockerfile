# Start from the official Airflow image that matches your docker-compose.yaml
FROM apache/airflow:3.0.1 

# Set the user to root temporarily if you need to install system dependencies
# (uncommon for pure Python packages, but good practice if needed)
USER root

# You can add system-level dependencies here if any of your Python packages require them.
# Example:
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Copy your requirements.txt from its 'app' subdirectory into the Docker image
COPY app/requirements.txt /requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# You can also copy your DAGs, plugins, and app code here to bake them into the image.
# This makes the image self-contained.
# COPY dags /opt/airflow/dags
# COPY plugins /opt/airflow/plugins
# COPY app /opt/airflow/app