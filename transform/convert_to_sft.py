import json
import os

class SFTConverter:
    def __init__(self, input_file, output_file, system_prompt):
        self.input_file = input_file
        self.output_file = output_file
        self.system_prompt = system_prompt

    def convert(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, \
                 open(self.output_file, 'w', encoding='utf-8') as outfile:
                
                for line in infile:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    prompt = data.get("enhanced_prompt", "")
                    completion = data.get("enhanced_completion", "")
                    
                    # Format to SFT messages array
                    sft_data = {
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion}
                        ]
                    }
                    
                    # Write out the new line
                    outfile.write(json.dumps(sft_data) + '\n')
                    
            print(f"Successfully converted {self.input_file} to SFT format: {self.output_file}")

        except Exception as e:
            print(f"Error during conversion: {e}")

if __name__ == "__main__":
    INPUT_FILE = "processing/ayurvedic_dataset_processed.jsonl"
    OUTPUT_FILE = "transform/ayurvedic_dataset_sft.jsonl"
    SYSTEM_PROMPT = "You are an expert Ayurvedic medical assistant. You provide accurate diagnoses and comprehensive Ayurvedic treatment plans based on patient symptoms, profiles, and medical history."

    converter = SFTConverter(INPUT_FILE, OUTPUT_FILE, SYSTEM_PROMPT)
    converter.convert()
