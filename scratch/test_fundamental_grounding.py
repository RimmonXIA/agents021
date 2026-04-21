import asyncio
from core.agents.synthesizer import AgentSynthesizer
from core.memory.blackboard import Blackboard

async def test_fundamental_grounding():
    # 1. Setup Blackboard (initializes WorldState)
    bb = Blackboard(session_id="test_fundamental", original_intent="Verify temporal awareness")
    
    # 2. Get Synthesizer
    asynth = AgentSynthesizer()
    
    # 3. Synthesize an agent (this should trigger _ground_agent middleware)
    # We pass the world_state data in the context
    context = {
        "current_date": bb.state.world_state.current_date,
        "current_time": bb.state.world_state.current_time
    }
    
    agent = asynth.synthesize("search", context)
    
    print("\n--- AGENT INSTRUCTIONS (Fundamental Grounding) ---")
    if isinstance(agent.instructions, list):
        print("\n".join(agent.instructions[:5])) # Show top few lines
    else:
        print(str(agent.instructions)[:500])
    print("--------------------------------------------------")
    
    # Check if the grounding block is present
    found = "### TRINITY SYSTEM GROUNDING ###" in (
        "\n".join(agent.instructions) if isinstance(agent.instructions, list) else agent.instructions
    )
    
    if found:
        print("SUCCESS: Fundamental grounding block found in agent instructions.")
    else:
        print("FAILURE: Grounding block NOT found.")

if __name__ == "__main__":
    asyncio.run(test_fundamental_grounding())
