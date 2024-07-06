import os
from huggingface_hub import hf_hub_download

def download_pth_from_huggingface(repo_id: str, filename: str, local_path: str):
    """
    Download a .pth file from a Hugging Face repository.

    Args:
        repo_id (str): The repository ID from which to download the file (e.g., "username/repo_name").
        filename (str): The name of the .pth file to download.
        local_path (str): The local path where the file will be saved.
    """
    try:
        # Download file from Hugging Face Hub
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(local_path))
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")

    # Check if file was downloaded
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found at {local_path}. Please check the repository and file name.")
