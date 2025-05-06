import kagglehub
import os
import shutil
import pandas as pd

database_dir = 'DataBaseFold'

def download_database():
    dataset_path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")


    os.makedirs(database_dir, exist_ok=True)

   
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            src = os.path.join(dataset_path, file)
            dst = os.path.join(database_dir, file)
            shutil.copy(src, dst)

    print("Download concluded.")

def import_database(dataset_name):
    database_path = os.path.join('DataBaseFolder', dataset_name)
    dataset = pd.read_csv(database_path)

    return dataset

if __name__ == "__main__":
    download_database()