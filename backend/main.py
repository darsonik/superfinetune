import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
# Add backend dir to path so we can import from ingest, transform, etc.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest.raw_data import data_ingest
from transform.jsonl_creator import JSONLCreator
from processing.adaption_processing import ProcessDataset
from adaption import Adaption, DatasetTimeout
from transform.convert_to_sft import SFTConverter
from finetune_job.pipeline import PioneerPipeline

from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fallback in-memory store if supabase not configured
jobs = {}

def set_job_state(job_id: str, data: dict):
    if supabase:
        supabase.table("jobs").update(data).eq("id", job_id).execute()
    else:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update(data)

def get_job_state(job_id: str):
    if supabase:
        res = supabase.table("jobs").select("*").eq("id", job_id).execute()
        if res.data:
            return res.data[0]
        return None
    return jobs.get(job_id)

def create_job_state(job_id: str, data: dict):
    if supabase:
        supabase.table("jobs").insert({"id": job_id, **data}).execute()
    else:
        jobs[job_id] = data

class PipelineRequest(BaseModel):
    kaggle_file_name: str
    kaggle_file_path: str
    prompt_mapping: Dict[str, str]
    completion_mapping: Dict[str, str]
    prompt_header: str = ""
    system_prompt: str = ""

class FinetuneRequest(BaseModel):
    dataset_name: str
    model_name: str
    
def run_pipeline_task(job_id: str, req: PipelineRequest):
    set_job_state(job_id, {"status": "running", "step": "ingesting", "message": "Downloading data from Kaggle"})
    
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = os.path.splitext(req.kaggle_file_name)[0]
        csv_file_path = os.path.join(backend_dir, "ingest", f"{base_name}.csv")
        jsonl_file_path = os.path.join(backend_dir, "transform", f"{base_name}.jsonl")
        
        os.makedirs(os.path.join(backend_dir, "ingest"), exist_ok=True)
        os.makedirs(os.path.join(backend_dir, "transform"), exist_ok=True)
        os.makedirs(os.path.join(backend_dir, "processing"), exist_ok=True)
        
        # 1. Ingest Data
        ingestor = data_ingest(req.kaggle_file_name, req.kaggle_file_path, csv_file_path)
        df = ingestor.ingest_data()
        df = ingestor.add_index(df)
        ingestor.save_csv(df, ingestor.csv_file_path)
        
        set_job_state(job_id, {"step": "creating_jsonl", "message": "Creating JSONL file"})
        
        # 2. Convert to JSONL
        creator = JSONLCreator(csv_file_path, jsonl_file_path)
        creator.generate(
            prompt_dict=req.prompt_mapping, 
            completion_dict=req.completion_mapping,
            prompt_header=req.prompt_header
        )
        
        set_job_state(job_id, {"step": "adaption_processing", "message": "Uploading to Adaption"})
        
        # 3. Process with Adaption
        client = Adaption(api_key=os.getenv("ADAPTION_API_KEY"))
        processor = ProcessDataset(client, jsonl_file_path)
        processor.upload_file()
        
        set_job_state(job_id, {"message": "Waiting for Adaption dataset processing..."})
        while True:
            row_count = processor.get_status()
            if row_count is not None:
                break
            time.sleep(2)
            
        run_response = processor.run()
        set_job_state(job_id, {"message": f"Running Adaption evaluation (Run ID: {run_response.run_id})..."})
        
        try:
            status = processor.wait_for_completion(timeout=5400)
            if status.status in ("failed", "skipped"):
                set_job_state(job_id, {"status": "failed", "message": "Adaption run failed."})
                return
        except Exception as e:
            set_job_state(job_id, {"status": "failed", "message": f"Adaption timeout: {str(e)}"})
            return
            
        url = processor.download()
        parsed_url = urllib.parse.urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        if not original_filename:
            original_filename = os.path.basename(jsonl_file_path)
            
        name, ext = os.path.splitext(original_filename)
        processed_filename = f"{name}_processed{ext}"
        processed_file_path = os.path.join(backend_dir, "processing", processed_filename)
        
        urllib.request.urlretrieve(url, processed_file_path)
        
        # 4. Create SFT file
        set_job_state(job_id, {"step": "creating_sft", "message": "Converting processed file to SFT format"})
        
        sft_file_path = os.path.join(backend_dir, "transform", f"{base_name}_sft.jsonl")
        
        converter = SFTConverter(processed_file_path, sft_file_path, req.system_prompt)
        converter.convert()
        
        set_job_state(job_id, {
            "status": "completed",
            "message": "Pipeline finished successfully",
            "files": {
                "raw_jsonl": jsonl_file_path,
                "processed_file": processed_file_path,
                "sft_jsonl": sft_file_path
            }
        })
    except Exception as e:
        set_job_state(job_id, {"status": "failed", "message": str(e)})

def run_finetune_task(job_id: str, req: FinetuneRequest):
    set_job_state(job_id, {"status": "running"})
    try:
        pipeline = PioneerPipeline()
        
        pipeline.upload_dataset("dummy_path", req.dataset_name)
        
        set_job_state(job_id, {"step": "finetuning", "message": "Started Pioneer Finetuning..."})
        
        pioneer_job_id = pipeline.train_model(req.model_name, req.dataset_name)
        
        # Poll pioneer job status
        while True:
            status_data = pipeline.get_job_status(pioneer_job_id)
            p_status = status_data.get("status")
            set_job_state(job_id, {"message": f"Pioneer Status: {p_status}"})
            
            if p_status == "complete":
                set_job_state(job_id, {
                    "status": "completed",
                    "message": "Finetuning finished",
                    "inference_endpoint": pioneer_job_id
                })
                break
            elif p_status in ("failed", "stopped"):
                set_job_state(job_id, {"status": "failed", "message": f"Pioneer job failed: {p_status}"})
                break
            
            time.sleep(10)
    except Exception as e:
        set_job_state(job_id, {"status": "failed", "message": str(e)})

@app.post("/api/pipeline/start")
def start_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job_state(job_id, {"status": "queued", "message": "Pipeline queued"})
    background_tasks.add_task(run_pipeline_task, job_id, req)
    return {"job_id": job_id}

@app.get("/api/pipeline/status/{job_id}")
def get_pipeline_status(job_id: str):
    data = get_job_state(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return data

@app.post("/api/pipeline/finetune")
def start_finetune(req: FinetuneRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job_state(job_id, {"status": "queued", "message": "Finetuning queued"})
    background_tasks.add_task(run_finetune_task, job_id, req)
    return {"job_id": job_id}

@app.get("/api/download")
def download_file(path: str):
    import fastapi.responses as responses
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return responses.FileResponse(full_path, filename=os.path.basename(full_path))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
