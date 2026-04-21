import asyncio
from core.agents.synthesizer import AgentSynthesizer
from core.memory.blackboard import Blackboard

async def test_developer():
    synth = AgentSynthesizer()
    bb = Blackboard(session_id="test_sid")
    
    print("Testing Developer Agent Synthesis...")
    try:
        agent = synth.synthesize("developer", {"additional_info": "Testing shell capability"})
        print(f"Agent Name: {agent.name}")
        print(f"Tools: {[t.__class__.__name__ for t in agent.tools]}")
        
        # We won't actually run it to save tokens/time, but we could.
        # response = agent.run("List the files in the current directory.")
        # print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"Synthesis failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_developer())
