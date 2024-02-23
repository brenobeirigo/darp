import os
import re
import json
import logging
import subprocess
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config():
    # Use pathlib for path operations
    config_file_path = Path.cwd() / "config.json"
    with config_file_path.open("r") as file:
        config = json.load(file)
    # Update paths to be absolute
    for key, value in config.items():
        # Join paths using pathlib
        config[key] = str(Path.cwd() / value)
    return config


def download_content(link, folder):
    """
    Downloads content from a given link to the specified folder.
    """

    # Example:
    # link = 'http://neumann.hec.ca/chairedistributique/data/darp/tabu/'
    # folders = chairedistributique/data/darp/tabu/
    # cut_dirs = 4 -> ignore the four first folders:
    # - chairedistributique
    # - data
    # - darp
    # - tabu
    folder_path = re.match(r"http://[^/]+/(.+)", link)
    cut_dirs = folder_path[1].count("/")

    command = [
        "wget",  # Download files using Unix-based systems
        "-r",  # Recursive download: Download the target URL and all links within
        "-np",  # No parent: Do not ascend to the parent directory when retrieving recursively
        "-nH",  # No host directories: Do not create directories for hostnames
        # --cut-dirs=NUMBER ignore NUMBER remote directory components
        f"--cut-dirs={cut_dirs}",  # Skip a certain number of directory levels in the URL
        "-P",  # Specify a prefix (directory) where files will be saved
        folder,  # The target directory where files will be saved
        link,  # The URL to download
    ]
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully downloaded {link} to {folder}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {link}. Error: {e}")


def process_csv(file_path, destination_folder):
    """
    Processes each row in the CSV file using pandas.
    """
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        author = row["First reference"].split()[0].lower()
        code = row["Code"].lower()
        year = re.findall(r"[0-9]+", row["First reference"])[0]
        link = row["Link"]

        folder = Path(destination_folder) / f"{code}_{author}_{year}"

        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.debug(f"Created directory {folder}")

        download_content(link, folder)


if __name__ == "__main__":
    config = load_config()
    process_csv(
        config["FILEPATH_DATA_SOURCE"], config["FOLDER_DATA_RAW_INSTANCES"]
    )
