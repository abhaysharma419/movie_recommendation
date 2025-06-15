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
