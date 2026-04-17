from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    hf_token = os.getenv("HF_TOKEN")
    dataset = load_dataset("cais/hle", split="test", token=hf_token)
    
    count_cs = 0
    count_cs_text = 0
    for item in dataset:
        if item.get("category") == "Computer Science/AI":
            count_cs += 1
            if item.get("image") is None:
                count_cs_text += 1
    
    print(f"Total CS: {count_cs}")
    print(f"Textual CS: {count_cs_text}")
    
    if count_cs > 0:
        first_cs = [x for x in dataset if x.get("category") == "Computer Science/AI"][0]
        print("\nFirst CS item keys:")
        print(first_cs.keys())
        print("\nImage field type:", type(first_cs.get("image")))
        print("Image field value:", first_cs.get("image"))

if __name__ == "__main__":
    main()
