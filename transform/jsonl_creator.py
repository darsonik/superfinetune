import pandas as pd
import json
import os
import re

class JSONLCreator:
    def __init__(self, csv_path, jsonl_path):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

    def get_val(self, row, col):
        """Helper to safely extract a value from the dataframe row, returning 'Not specified' if NaN/empty."""
        if col not in row:
            return "Not specified"
        val = row[col]
        if pd.isna(val) or val == "":
            return "Not specified"
        return str(val).strip()

    def _build_text(self, row, mapping_dict):
        """
        Builds a text block from the mapping dictionary.
        If the dictionary value contains curly braces (e.g., "{Age} | {Gender}"), 
        it will format the string using columns. 
        Otherwise, it assumes the value is a direct column name.
        """
        lines = []
        for label, template in mapping_dict.items():
            if "{" not in template:
                # Standard 1-to-1 mapping
                val = self.get_val(row, template)
                if label:
                    lines.append(f"{label}: {val}")
                else:
                    lines.append(val)
            else:
                # Template formatting (e.g., "Stress - {Stress Levels}, Sleep - {Sleep Patterns}")
                cols_in_template = re.findall(r'\{(.*?)\}', template)
                formatted_str = template
                for col in cols_in_template:
                    formatted_str = formatted_str.replace(f"{{{col}}}", self.get_val(row, col))
                
                if label:
                    lines.append(f"{label}: {formatted_str}")
                else:
                    lines.append(formatted_str)
                    
        return "\n".join(lines)

    def generate(self, prompt_dict, completion_dict, prompt_header="", completion_header=""):
        """Reads CSV and generates JSONL using the provided dictionaries."""
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        
        print(f"Reading {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_path}. Please make sure you are in the project root.")
            return

        records = []
        for _, row in df.iterrows():
            # Build Prompt
            prompt_content = self._build_text(row, prompt_dict)
            prompt = f"{prompt_header}\n{prompt_content}" if prompt_header else prompt_content
            
            # Build Completion
            completion_content = self._build_text(row, completion_dict)
            completion = f"{completion_header}\n{completion_content}" if completion_header else completion_content
            
            records.append({
                "prompt": prompt.strip(),
                "completion": completion.strip()
            })
            
        print(f"Writing {len(records)} records to {self.jsonl_path}...")
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print("Done! JSONL dataset is ready.")

if __name__ == "__main__":
    # Example usage using the new generalized method!
    csv_path = "ingest/ayurvedic_dataset.csv"
    jsonl_path = "transform/ayurvedic_dataset.jsonl"
    
    # 1. Define the mappings for the Prompt
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

    # 2. Define the mappings for the Completion
    completion_mapping = {
        "Diagnosis": "{Disease} (Hindi: {Hindi Name}, Marathi: {Marathi Name})",
        "Dosha Imbalance": "Doshas",
        "Prakriti (Constitution)": "Constitution/Prakriti",
        "": "\n--- Ayurvedic Treatment Plan ---", # Empty key acts as a spacer/header
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

    creator = JSONLCreator(csv_path, jsonl_path)
    creator.generate(
        prompt_dict=prompt_mapping, 
        completion_dict=completion_mapping,
        prompt_header=prompt_header
    )
