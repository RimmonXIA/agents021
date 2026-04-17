import os
import json
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from core.meso_plane.fsm_workflow import MacroMesoMicroWorkflow

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate HLE dataset")
    parser.add_argument("--num", "-n", type=int, default=3, help="Number of test cases to evaluate")
    args = parser.parse_args()
    
    print("Loading HLE dataset...")
    # Load the HLE dataset
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\nWARNING: 'cais/hle' is a gated dataset.")
        print("Please accept the terms at https://huggingface.co/datasets/cais/hle")
        print("Then add your HF_TOKEN to your .env file.\n")
        
    try:
        dataset = load_dataset("cais/hle", split="test", token=hf_token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have accepted the dataset conditions and set HF_TOKEN in your .env")
        return
    
    # Target filters: CS and Finance/Economics
    def hle_filter(x):
        is_cs = (x.get("category") == "Computer Science/AI")
        is_finance = (x.get("raw_subject") in ["Economics", "Finance"])
        is_textual = not x.get("image")
        return (is_cs or is_finance) and is_textual
    
    filtered_ds = dataset.filter(hle_filter)
    
    print(f"Found {len(filtered_ds)} matching textual questions.")
    
    # Limit to requested number for testing
    num_tests = min(args.num, len(filtered_ds))
    test_subset = filtered_ds.select(range(num_tests))
    
    results = []
    
    os.makedirs("state", exist_ok=True)
    
    workflow = MacroMesoMicroWorkflow(session_id="hle-eval-001")
    
    for i, item in enumerate(test_subset):
        question = item["question"]
        answer = item["answer"]
        subject = item.get("raw_subject", "Unknown")
        
        print(f"\n[{i+1}/{num_tests}] Evaluating ({subject})...")
        print(f"Question: {question}")
        
        response = workflow.run(question)
        
        # Verbose output from telemetry
        events = workflow.session_state.get("telemetry_events", [])
        for event in events:
            etype = event.get("event_type")
            payload = event.get("payload", {})
            if etype == "MACRO_PLAN_CREATED":
                print(f"\n--- Macro Plan ---\n{json.dumps(payload.get('plan'), indent=2)}")
            elif etype == "TASK_COMPLETED":
                print(f"Task Complete: {payload.get('capability')}")
            elif etype == "TASK_ERROR":
                print(f"Task Error: {payload.get('capability')} - {payload.get('error')}")
            elif etype == "MACRO_ERROR":
                print(f"Macro Error: {payload.get('error')}")
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"\nFinal Response:\n{content}")
        print("-" * 50)
        
        results.append({
            "id": item.get("id", f"q_{i}"),
            "question": question,
            "expected_answer": answer,
            "subject": subject,
            "predicted_content": content,
            "telemetry": workflow.session_state.get("telemetry_events", [])
        })
        
        # Reset telemetry for next run
        workflow.session_state["telemetry_events"] = []

    with open("hle_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nEvaluation complete. Results saved to hle_results.json")

if __name__ == "__main__":
    main()
