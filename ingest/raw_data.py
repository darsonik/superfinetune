import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# (Make sure KAGGLE_USERNAME and KAGGLE_KEY are in your .env)
load_dotenv()

class data_ingest:
    def __init__(self, kaggle_file_name, kaggle_file_path, csv_file_path):
        self.kaggle_file_name = kaggle_file_name
        self.kaggle_file_path = kaggle_file_path
        self.csv_file_path = csv_file_path

    def ingest_data(self):
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            self.kaggle_file_path,
            self.kaggle_file_name,
        )
        return df   

    def add_index(self, df):
        df["id"] = range(len(df))
        return df
    
    def save_csv(self, df, path):
        df.to_csv(path, index=False)

if __name__ == "__main__":
    kaggle_file_name = "AyurGenixAI_Dataset.csv"
    kaggle_file_path = "kagglekirti123/ayurgenixai-ayurvedic-dataset"
    csv_file_path = "ingest/ayurvedic_dataset.csv"
    ingestor = data_ingest(kaggle_file_name, kaggle_file_path, csv_file_path)
    df = ingestor.ingest_data()
    df = ingestor.add_index(df)
    ingestor.save_csv(df, ingestor.csv_file_path)
    print(f"Done! Saved to {ingestor.csv_file_path}")