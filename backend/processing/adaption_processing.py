import time
from adaption import DatasetTimeout
from adaption import Adaption
from dotenv import load_dotenv
import os
import sys
import urllib.request
import urllib.parse

load_dotenv()

class ProcessDataset:
    def __init__(self, client, path):
        self.client = client
        self.path = path
        self.dataset_id = None

    def upload_file(self):
        result = self.client.datasets.upload_file(path=self.path)
        self.dataset_id = result.dataset_id
        return self.dataset_id

    def get_status(self):
        status = self.client.datasets.get_status(self.dataset_id)
        return status.row_count

    def estimate_run(self):
        estimate = self.client.datasets.run(
            self.dataset_id,
            estimate=True,
            column_mapping={"prompt":"prompt",
            "completion": "completion"},
            brand_controls={"hallucination_mitigation": True},
        )
        return estimate

    def run(self):
        run = self.client.datasets.run(
            self.dataset_id,
            column_mapping={"prompt":"prompt",
            "completion": "completion"},
            brand_controls={"hallucination_mitigation": True},
        )
        return run

    def wait_for_completion(self, timeout=3600):
        status = self.client.datasets.wait_for_completion(self.dataset_id, timeout=timeout)
        return status

    def get_evaluation(self):
        evaluation = self.client.datasets.get_evaluation(self.dataset_id)
        return evaluation

    def download(self):
        url = self.client.datasets.download(self.dataset_id)
        return url

if __name__ == "__main__":
    client = Adaption(api_key=os.getenv("ADAPTION_API_KEY"))
    path = "ayurvedic_dataset_sft.jsonl"
    processor = ProcessDataset(client, path)
    
    print("Uploading file...")
    processor.upload_file()
    print(f"Dataset ID: {processor.dataset_id}")
    
    print("Waiting for dataset to be processed...")
    while True:
        row_count = processor.get_status()
        if row_count is not None:
            break
        time.sleep(2)
        
    estimate = processor.estimate_run()
    print(f"Would cost {estimate.estimated_credits_consumed} credits")
    
    run_response = processor.run()
    print(f"Run ID: {run_response.run_id}")
    print(f"Estimated credits: {run_response.estimated_credits_consumed}")
    
    try: 
        status = processor.wait_for_completion(timeout=5400)
        if status.status in ("failed", "skipped"):
            print(f"Finished: {status.status}")
            sys.exit(1)
        print(f"Finished: {status.status}")
    except DatasetTimeout as e:
        print(f"Timed out: {e}")
        sys.exit(1)
        
    while True:
        evaluation = processor.get_evaluation()
        if evaluation.status in ("succeeded", "failed", "skipped"):
            break
        time.sleep(5)
        
    if evaluation.status == "succeeded" and evaluation.quality:
        print(evaluation.quality.model_dump(exclude_none=True))
        
    url = processor.download()
    print(f"Download URL: {url}")
    
    parsed_url = urllib.parse.urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    if not original_filename:
        original_filename = os.path.basename(path)
        
    name, ext = os.path.splitext(original_filename)
    new_filename = f"{name}_processed{ext}"
    
    save_dir = "processing"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, new_filename)
    
    print(f"Downloading file to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download completed.")