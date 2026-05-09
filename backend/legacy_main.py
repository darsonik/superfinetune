import os
import sys
import time
import urllib.parse
import urllib.request
from dotenv import load_dotenv

# Import the necessary classes from our modules
from ingest.raw_data import data_ingest
from transform.jsonl_creator import JSONLCreator
from processing.adaption_processing import ProcessDataset
from adaption import Adaption, DatasetTimeout
from transform.convert_to_sft import SFTConverter

def main():
    load_dotenv()
    
    # Inputs
    kaggle_file_name = "AyurGenixAI_Dataset.csv"
    kaggle_file_path = "kagglekirti123/ayurgenixai-ayurvedic-dataset"
    
    # Dynamically define file paths based on the kaggle file name
    base_name = os.path.splitext(kaggle_file_name)[0]
    csv_file_path = f"ingest/{base_name}.csv"
    jsonl_file_path = f"transform/{base_name}.jsonl"
    
    # Create required directories
    os.makedirs("ingest", exist_ok=True)
    os.makedirs("transform", exist_ok=True)
    os.makedirs("processing", exist_ok=True)

    # 1. Raw data ingested via raw_data.py
    print(f"\n--- STEP 1: Ingesting Data ---")
    ingestor = data_ingest(kaggle_file_name, kaggle_file_path, csv_file_path)
    df = ingestor.ingest_data()
    df = ingestor.add_index(df)
    ingestor.save_csv(df, ingestor.csv_file_path)
    print(f"Data ingested and saved to {csv_file_path}")

    # 2. Convert to JSONL using jsonl_creator.py
    print(f"\n--- STEP 2: Creating JSONL ---")
    # Using the hardcoded prompt and column mapping
    prompt_mapping = {
        "Symptoms": "Symptoms",
        "Severity": "Symptom Severity",
        "Patient Profile": "{Age Group} | {Gender}",
        "Medical History": "Medical History",
        "Current Medications": "Current Medications",
        "Risk Factors": "Risk Factors",
        "Environmental Factors": "Environmental Factors",
        "Lifestyle & Diet": "{Occupation and Lifestyle} | Diet: {Dietary Habits}",
        "Stress & Sleep": "Stress - {Stress Levels}, Sleep - {Sleep Patterns}"
    }

    completion_mapping = {
        "Diagnosis": "{Disease} (Hindi: {Hindi Name}, Marathi: {Marathi Name})",
        "Dosha Imbalance": "Doshas",
        "Prakriti (Constitution)": "Constitution/Prakriti",
        "": "\n--- Ayurvedic Treatment Plan ---",
        "Herbs": "Ayurvedic Herbs",
        "Formulation": "Formulation",
        "Diet & Lifestyle Recommendations": "Diet and Lifestyle Recommendations",
        "Yoga & Physical Therapy": "Yoga & Physical Therapy",
        "": "\n--- Additional Guidance ---",
        "General Recommendations": "Patient Recommendations",
        "Prevention": "Prevention",
        "Prognosis": "Prognosis"
    }
    prompt_header = "Based on the following patient profile, symptoms, and medical history, provide an accurate diagnosis and a comprehensive Ayurvedic treatment plan.\n\n--- Patient Details ---"
    
    creator = JSONLCreator(csv_file_path, jsonl_file_path)
    creator.generate(
        prompt_dict=prompt_mapping, 
        completion_dict=completion_mapping,
        prompt_header=prompt_header
    )
    print(f"JSONL created at {jsonl_file_path}")

    # 3. Process it using adaption_processing.py and download the processed file
    print(f"\n--- STEP 3: Processing Dataset with Adaption ---")
    client = Adaption(api_key=os.getenv("ADAPTION_API_KEY"))
    processor = ProcessDataset(client, jsonl_file_path)
    
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
        print(f"Finished: {status.status}")
        if status.status in ("failed", "skipped"):
            print("Run failed or skipped. Exiting.")
            sys.exit(1)
    except DatasetTimeout as e:
        print(f"Timed out: {e}")
        sys.exit(1)
        
    # while True:
    #     evaluation = processor.get_evaluation()
    #     if evaluation.status in ("succeeded", "failed", "skipped"):
    #         break
    #     time.sleep(5)
        
    # if evaluation.status == "succeeded" and evaluation.quality:
    #     print(evaluation.quality.model_dump(exclude_none=True))
        
    url = processor.download()
    print(f"Download URL: {url}")
    
    parsed_url = urllib.parse.urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    if not original_filename:
        original_filename = os.path.basename(jsonl_file_path)
        
    name, ext = os.path.splitext(original_filename)
    processed_filename = f"{name}_processed{ext}"
    processed_file_path = os.path.join("processing", processed_filename)
    
    print(f"Downloading file to {processed_file_path}...")
    urllib.request.urlretrieve(url, processed_file_path)
    print("Download completed.")

    # 4. Create a SFT file from the processed file using convert_to_sft.py
    print(f"\n--- STEP 4: Converting to SFT Format ---")
    sft_file_path = f"transform/{base_name}_sft.jsonl"
    system_prompt = "You are an expert Ayurvedic medical assistant. You provide accurate diagnoses and comprehensive Ayurvedic treatment plans based on patient symptoms, profiles, and medical history."
    
    converter = SFTConverter(processed_file_path, sft_file_path, system_prompt)
    converter.convert()
    print(f"Orchestration complete! SFT file saved to {sft_file_path}")

if __name__ == "__main__":
    main()
