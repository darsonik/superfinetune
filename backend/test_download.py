import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from processing.adaption_processing import ProcessDataset
from adaption import Adaption

load_dotenv()
client = Adaption(api_key=os.getenv("ADAPTION_API_KEY"))
processor = ProcessDataset(client, "")
processor.dataset_id = "217aecbc-d708-4e56-96f5-ee808bcb088f"

print(f"Fetching download URL for dataset {processor.dataset_id}...")
url = processor.download()
print(f"Download URL: {url}")

parsed_url = urllib.parse.urlparse(url)
original_filename = os.path.basename(parsed_url.path)
if not original_filename:
    original_filename = "dataset.jsonl"
    
name, ext = os.path.splitext(original_filename)
processed_filename = f"{name}_processed{ext}"
processed_file_path = os.path.join("processing", processed_filename)

print(f"Downloading file to {processed_file_path}...")
urllib.request.urlretrieve(url, processed_file_path)
print("Download completed.")
