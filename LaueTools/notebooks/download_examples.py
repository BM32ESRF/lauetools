import os
import sys
import requests
from pathlib import Path

# Define the files to download (update these URLs with your actual GitHub Releases links)
EXAMPLE_FILES = {
    "notebooks/PeakSearch_MultiProcessing.ipynb": "https://github.com/BM32ESRF/lauetools/releases/download/3.2.10-data/PeakSearch_MultiProcessing.ipynb",
    "notebooks/Indexation_MultiProcessing_april2025.ipynb": "https://github.com/BM32ESRF/lauetools/releases/download/3.2.10-data/Indexation_MultiProcessing_april2025.ipynb",
    "LaueImages/AH12_CMT_r14_0000.tif": "https://github.com/BM32ESRF/lauetools/releases/download/v3.2.8-data/AH12_CMT_r14_0000.tif",
    # Add more files as needed
}

def download_examples(save_dir="."):
    """Download example notebooks and data to the specified directory."""
    save_dir = Path(save_dir)

    for filename, url in EXAMPLE_FILES.items():
        save_path = save_dir / filename
        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not save_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"{filename} already exists. Skipping.")

    print(f"Examples downloaded to: {save_dir.absolute()}")

if __name__ == "__main__":
    # Use the first command-line argument as save_dir, or default to current directory
    save_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    download_examples(save_dir)