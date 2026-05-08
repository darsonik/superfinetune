import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

PIONEER_API_KEY = os.getenv("PIONEER_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL")
PIONEER_API_URL = "https://api.pioneer.ai"

class PioneerPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or PIONEER_API_KEY
        if not self.api_key:
            raise ValueError("PIONEER_API_KEY is required.")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def upload_dataset(self, file_path: str, dataset_name: str) -> str:
        """
        Dummy method to upload a dataset. 
        In the future, this should upload the processed JSONL file to Pioneer.
        """
        print(f"Dummy: Uploading dataset from {file_path} as '{dataset_name}'...")
        time.sleep(1) # Simulate upload time
        print(f"Dummy: Dataset '{dataset_name}' uploaded successfully.")
        return dataset_name

    def train_model(self, model_name: str, dataset_name: str, base_model: str = None, 
                    training_type: str = "lora", nr_epochs: int = 5, learning_rate: float = 5e-5) -> str:
        """
        Starts a training job with the given dataset and base model.
        Returns the job_id.
        """
        base_model = base_model or BASE_MODEL
        print(f"Starting training job for model '{model_name}' using base model '{base_model}' and dataset '{dataset_name}'...")
        
        url = f"{PIONEER_API_URL}/felix/training-jobs"
        payload = {
            "model_name": model_name,
            "base_model": base_model,
            "datasets": [{"name": dataset_name}],
            "training_type": training_type,
            "nr_epochs": nr_epochs,
            "learning_rate": learning_rate
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        job_id = data.get("id")
        print(f"Training job started successfully. Job ID: {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """
        Gets the status and metrics of a training job.
        """
        url = f"{PIONEER_API_URL}/felix/training-jobs/{job_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def run_inference(self, model_id: str, text: str, schema: dict = None, task: str = None) -> dict:
        """
        Runs inference against a fine-tuned or base model.
        """
        print(f"Running inference on model '{model_id}'...")
        url = f"{PIONEER_API_URL}/inference"
        
        payload = {
            "model_id": model_id,
            "text": text,
            "threshold": 0.5
        }
        
        if schema is not None:
            payload["schema"] = schema
        elif task is not None:
            payload["task"] = task
        else:
            payload["task"] = "generate"
            
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("Inference completed.")
        return result

if __name__ == "__main__":
    # Example usage:
    pipeline = PioneerPipeline()
    
    # 1. Upload dataset
    dataset_name = pipeline.upload_dataset("path/to/processed_data.jsonl", "my-training-dataset")
    
    # 2. Train model (uncomment to run)
    # job_id = pipeline.train_model("my-finetuned-model", dataset_name)
    
    # 3. Check job status (uncomment to run)
    # status = pipeline.get_job_status(job_id)
    # print(status)

    # 4. Inference (example with base model since training takes time)
    # response = pipeline.run_inference(
    #     model_id=BASE_MODEL,
    #     text="Tell me about the benefits of Ashwagandha.",
    #     task="generate"
    # )
    # print(response)
