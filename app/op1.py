import requests
import zipfile
import os

def download_and_extract_zip(url, download_dir, extract_dir):
    """
    Downloads a ZIP file from a given URL and extracts its contents.

    Args:
        url (str): The URL of the ZIP file.
        download_dir (str): The directory where the ZIP file will be temporarily saved.
        extract_dir (str): The directory where the contents of the ZIP file will be extracted.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created download directory: {download_dir}")

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"Created extraction directory: {extract_dir}")

    zip_filename = os.path.join(download_dir, url.split('/')[-1])
    
    print(f"Downloading {url} to {zip_filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return False

    print(f"Extracting {zip_filename} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: {zip_filename} is not a valid ZIP file.")
        return False
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False
    finally:
        # Optionally, remove the downloaded zip file after extraction
        # os.remove(zip_filename)
        # print(f"Removed temporary zip file: {zip_filename}")
        pass # Keeping it for now for verification

    return True

if __name__ == "__main__":
    movielens_url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    download_directory = "temp_downloads" # Temporary directory to save the zip
    extraction_directory = "ml-1m_dataset" # Directory where the content will be extracted

    success = download_and_extract_zip(movielens_url, download_directory, extraction_directory)

    if success:
        print(f"\nMovieLens 1M dataset should now be in: {extraction_directory}")
        # You can now proceed to load data from 'ml-1m_dataset/ml-1m/'
        # List contents to verify
        print("\nContents of extraction directory:")
        for root, dirs, files in os.walk(extraction_directory):
            level = root.replace(extraction_directory, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{subindent}{f}')
    else:
        print("Failed to download and extract the dataset.")