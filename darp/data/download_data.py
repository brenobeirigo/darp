import csv
import re
import os


FOLDER_DATA_RAW_INSTANCES = os.path.join(
    os.getcwd(), "data","raw", "darp_instances")

FILEPATH_DATA_SOURCE = os.path.join(
    os.getcwd(), "darp", "data", "benchmark_instances.csv")

with open(FILEPATH_DATA_SOURCE, newline='') as csvfile:
    instances = csv.DictReader(csvfile)
    for row in instances:
        author = row["First reference"].split()[0].lower()
        code = row["Code"].lower()
        year = re.findall("[0-9]+", row["First reference"])[0]
        link = row["Link"]

        folder = os.path.join(
            FOLDER_DATA_RAW_INSTANCES,
            f"{code}_{author}_{year}")
        # -r: recursively download content
        # nH: no host
        # cut-dirs: cut the folder path until last folder
        cut_dirs = len(link.split("//")[1].split("/")) - 2
        # -P: create target folder
        command = f"wget -r -np -nH --cut-dirs={cut_dirs} {link} -P {folder}"
        print(f"Downloading {link} to {folder} ({command})...")

        os.system(command)
