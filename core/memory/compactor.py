import asyncio
from typing import Dict, Any, List
from core.agents.synthesizer import AgentSynthesizer
from core.utils.logging import get_logger

logger = get_logger(__name__)

class SemanticCompactor:
    """
    Manages the tiering of memory in the Blackboard.
    Compresses low-level task results into high-level semantic summaries.
    """
    def __init__(self, synthesizer: AgentSynthesizer):
        self.asynth = synthesizer
        self.threshold = 10 # Number of task results before compaction triggers

    async def compact(self, shared_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw shared memory, identifies compaction candidates,
        and returns a compacted dictionary.
        """
        task_result_keys = [k for k in shared_memory.keys() if k.endswith("_result")]
        
        if len(task_result_keys) < self.threshold:
            return shared_memory

        logger.info(f"Triggering Semantic Compaction for {len(task_result_keys)} keys...")
        
        # 1. Prepare data for distillation
        raw_data = {k: shared_memory[k] for k in task_result_keys}
        
        # 2. Synthesize Compactor Agent
        agent = self.asynth.synthesize("compactor", {})
        
        # 3. Distill
        prompt = f"Distill the following task results into a single technical summary:\n\n{raw_data}"
        try:
            # We use a simplified run here for speed
            response = await asyncio.to_thread(agent.run, prompt)
            summary = response.content
            
            # 4. Reconstruct Memory
            new_memory = {k: v for k, v in shared_memory.items() if k not in task_result_keys}
            
            # Add session summary
            existing_summary = shared_memory.get("session_summary", "")
            new_memory["session_summary"] = f"{existing_summary}\n\n### Batch Summary ###\n{summary}".strip()
            
            logger.info("Compaction completed successfully.")
            return new_memory
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            return shared_memory
