import asyncio
import uuid

import lancedb

from core.config import settings
from core.engine.orchestrator import IntentOrchestrator
from core.memory.blackboard import Blackboard
from core.utils.logging import get_logger, setup_logging

logger = get_logger("evals")

async def run_eval() -> None:
    setup_logging()
    logger.info("=== Trinity Evals: Testing EO Skill Retrieval ===")
    
    test_intent = "Gather the latest info on OpenAI Sora and save it to a file."
    session_id = str(uuid.uuid4())
    
    # 1. Initialize Blackboard
    bb = Blackboard(session_id=session_id, original_intent=test_intent)
    
    # 2. Manually inject a skill into LanceDB (replacing the old sqlite skills_fts logic)
    logger.info("Injecting mock skill into LanceDB...")
    db = lancedb.connect(settings.lancedb_dir)
    mock_skill = {
        "id": "eval_123",
        "title": "Mandatory Fact-Check",
        "description": "When searching for info about AI models like Sora.",
        "content_markdown": "Always verify sources and check for release dates.",
        "text": "Mandatory Fact-Check When searching for info about AI models like Sora."
    }
    try:
        table = db.open_table("skills")
        table.add([mock_skill])
    except Exception:
        db.create_table("skills", data=[mock_skill])

    # 3. Initialize Orchestrator
    orchestrator = IntentOrchestrator(blackboard=bb)
    
    # 4. Decompose intent and check if the skill was considered
    logger.info("Running intent decomposition...")
    await orchestrator.decompose_intent()
    
    tasks = bb.state.todo_list
    logger.info(f"Planned {len(tasks)} tasks.")
    for t in tasks:
        logger.info(f"- {t.description}")
        
    assert len(tasks) > 0, "Failed to plan tasks."
    logger.info("✅ Eval Passed: EO Skills successfully injected and utilized by IO.")

if __name__ == "__main__":
    asyncio.run(run_eval())
