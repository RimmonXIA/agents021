import os
import sys
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Ensure state directory exists for SQLite
    os.makedirs("state", exist_ok=True)
    
    from core.meso_plane.fsm_workflow import MacroMesoMicroWorkflow
    
    workflow = MacroMesoMicroWorkflow(session_id="run-001")
    response = workflow.run("Search the web for Agno frameworks, extract key features, and analyze our code for complexity.")
    
    print("\n--- Pipeline Result ---")
    print(response.content if hasattr(response, 'content') else response)
    
    print("\n--- Telemetry Log ---")
    import json
    print(json.dumps(workflow.session_state["telemetry_events"], indent=2))

if __name__ == "__main__":
    main()
