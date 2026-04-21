import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.orchestrator import IntentOrchestrator
from core.memory.blackboard import Blackboard
from core.utils.logging import get_logger

logger = get_logger("e2e_test")

async def run_test():
    print("\n" + "="*60)
    print("TRINITY E2E STABILITY TEST")
    print("="*60)
    
    session_id = "test_session_e2e"
    intent = "Research the current sentiment about AI agents in 2026 and create a one-line summary."
    
    # Clean up old data if any
    db_path = "data/trinity_memory.db"
    if os.path.exists(db_path):
        # We keep it to test sqlite persistence, but we'll use a new session
        pass

    bb = Blackboard(session_id=session_id, original_intent=intent)
    orchestrator = IntentOrchestrator(blackboard=bb)
    
    print(f"Testing Intent: {intent}")
    print("Starting Orchestration Loop...")
    
    try:
        # Run the loop
        # This will trigger decomposition and execution
        await asyncio.wait_for(orchestrator.run_loop(), timeout=300)
        
        print("\n" + "="*60)
        print("VERIFICATION PHASE")
        print("="*60)
        
        # 1. Check Status
        print(f"Loop Status: {bb.state.status}")
        assert bb.state.status == "completed", f"Expected 'completed', got {bb.state.status}"
        
        # 2. Check Tasks
        print(f"Completed Tasks: {len(bb.state.completed_tasks)}")
        assert len(bb.state.completed_tasks) > 0, "No tasks were completed!"
        
        # 3. Check Shared Memory
        print(f"Shared Memory Keys: {list(bb.state.shared_memory.keys())}")
        # We expect at least one result key
        result_keys = [k for k in bb.state.shared_memory.keys() if "_result" in k]
        print(f"Result Keys Found: {result_keys}")
        assert len(result_keys) > 0, "No results were saved to shared memory!"
        
        # 4. Check Trajectory
        print(f"Trajectory Steps: {len(bb.state.trajectory)}")
        assert len(bb.state.trajectory) == len(bb.state.completed_tasks), "Trajectory count mismatch!"
        
        print("\n[SUCCESS] Trinity Agentic Loop is stable and functional!")
        
    except asyncio.TimeoutError:
        print("\n[ERROR] Test timed out! Loop might be stuck.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Test encountered an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_test())
