import asyncio
import json
import re
from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from pydantic import BaseModel
from core.agents.synthesizer import AgentSynthesizer
from core.config import settings
from core.engine.optimizer import EvolutionaryOptimizer
from core.memory.blackboard import Blackboard
from core.models import AtomicTask, IOPlan, SubAgentResult, TrajectoryStep
from core.utils.logging import get_logger

logger = get_logger(__name__)

class IntentOrchestrator:
    """
    The Intent Orchestrator (IO) - The "nO Master Loop".
    Single-threaded execution loop managing state and sub-agent synthesis.
    """
    def __init__(self, blackboard: Blackboard):
        self.bb = blackboard
        self.asynth = AgentSynthesizer()
        self.planner: Agent = self.asynth.synthesize('planner', {})
        self.step_counter = 0
        self.running_tasks: dict[str, asyncio.Task[None]] = {} # task_id -> asyncio.Task
        self.concurrency_limit = asyncio.Semaphore(settings.max_concurrency if hasattr(settings, "max_concurrency") else 5)

    async def _safe_run(self, agent: Agent, prompt: str, response_model: type | None = None, max_retries: int | None = None) -> Any:
        """
        Centrally handles agent execution with retries, fallback models, and robust parsing.
        Includes self-correction for JSON parsing errors.
        """
        max_retries = max_retries or settings.max_retries
        current_attempt = 0
        last_error = None
        current_prompt = prompt
        
        while current_attempt < max_retries:
            current_attempt += 1
            try:
                if current_attempt > 1:
                    logger.warning(f"Retry attempt {current_attempt}/{max_retries} for {agent.name}...")
                    # Append feedback if it was a parsing error
                    if last_error and "parse" in str(last_error).lower():
                        current_prompt = prompt + f"\n\n### ERROR FROM PREVIOUS ATTEMPT ###\nYour response was not valid JSON or did not match the schema.\nError: {last_error}\nPlease try again and ensure strict JSON compliance."

                response = await asyncio.to_thread(agent.run, current_prompt, output_schema=response_model)
                content = response.content
                
                if hasattr(response, "reasoning_content") and response.reasoning_content:
                    logger.info(f"[{agent.name} Thinking Process]\n{response.reasoning_content}")

                if not content:
                    raise ValueError("Received empty response from model.")

                # Parsing Logic
                if response_model:
                    # Fast path: Check if agno already returned the correct type or a Pydantic model
                    if isinstance(content, (response_model, BaseModel)):
                        try:
                            # If it's already the right model, return it. 
                            # If it's a different BaseModel, try to validate it into our target.
                            if isinstance(content, response_model):
                                return content
                            return response_model.model_validate(content.model_dump())
                        except Exception as e:
                            logger.debug(f"Pydantic re-validation failed: {e}")

                    # agno may return a dict when output_schema is used
                    if isinstance(content, dict):
                        try:
                            return response_model.model_validate(content)
                        except Exception as e:
                            raise ValueError(f"Could not validate dict into {response_model.__name__}. Error: {e}")

                    # For string content: extract <thinking> blocks first, then parse JSON
                    if isinstance(content, str):
                        thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL | re.IGNORECASE)
                        if thinking_match:
                            thought = thinking_match.group(1).strip()
                            logger.info(f"[{agent.name} Internal Thought]\n{thought}")
                            content = content.replace(thinking_match.group(0), "").strip()

                        from agno.utils.string import parse_response_model_str

                        # Pre-clean: Extract from markdown if present
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if not json_match:
                            json_match = re.search(r'(\{.*\})', content, re.DOTALL)

                        target_str = json_match.group(1) if json_match else content

                        try:
                            parsed = parse_response_model_str(target_str, response_model)
                            if parsed:
                                return parsed
                        except Exception as parse_err:
                            logger.debug(f"Internal parse error: {parse_err}")

                        # Last ditch: direct json.loads + pydantic
                        try:
                            data = json.loads(target_str)
                            return response_model.model_validate(data)
                        except Exception as e:
                            raise ValueError(f"Could not parse response into {response_model.__name__}. Error: {e}")
                    else:
                        raise ValueError(f"Expected string or Pydantic model for parsing, got {type(content)}")

                return content

            except Exception as e:
                last_error = e
                logger.error(f"{agent.name} attempt {current_attempt} failed: {e}")
                
                if agent.model and hasattr(agent.model, "id") and agent.model.id == "deepseek-reasoner":
                    logger.info("Falling back to deepseek-chat for stability...")
                    agent.model = DeepSeek(id="deepseek-chat")

                if current_attempt >= max_retries:
                    break
        
        return last_error

    async def decompose_intent(self) -> None:
        """Uses the planner agent to break down the original intent."""
        logger.info(f"Decomposing intent: {self.bb.state.original_intent}")
        
        skills = self.bb.fetch_relevant_skills()
        skills_context = ""
        if skills:
            skills_context = "\n\n### Applicable Skills (SOPs) ###\n"
            for s in skills:
                skills_context += f"- **{s['title']}**: {s['description']}\n  Content: {s['content']}\n"
                
        prompt = self.bb.state.original_intent + skills_context
        plan = await self._safe_run(self.planner, prompt, response_model=IOPlan)
        if isinstance(plan, IOPlan):
            for task in plan.tasks:
                await self.bb.add_todo(task)
            logger.info(f"Added {len(plan.tasks)} tasks to TODO list.")
        else:
            logger.error(f"Failed to decompose intent: {plan}")
            await self.bb.mark_failed()

    async def run_loop(self) -> None:
        """
        The main orchestration loop. Executes until TODO is empty or a critical error occurs.
        Uses a concurrent branch-and-merge strategy.
        """
        await self.decompose_intent()
        
        # self.running_tasks is now an instance attribute

        while self.bb.state.status == "running":
            # 1. Get ready tasks from scheduler
            ready_tasks = await self.bb.get_ready_tasks()
            
            # 2. Launch ready tasks
            for task in ready_tasks:
                if task.id not in self.running_tasks:
                    logger.info(f"Scheduling Task: {task.description} (ID: {task.id})")
                    self.running_tasks[task.id] = asyncio.create_task(self._execute_task(task, self.running_tasks))

            # 3. Handle loop termination or waiting
            if not self.running_tasks:
                async with self.bb.lock:
                    has_todo = len(self.bb.state.todo_list) > 0
                
                if not has_todo:
                    logger.info("All tasks completed. Execution finished.")
                    await self.bb.mark_completed()
                else:
                    logger.error("No ready tasks but TODO list is not empty. Potential deadlock/circular dependency.")
                    await self.bb.mark_failed()
                break

            # 4. Wait for any task to finish
            done, _ = await asyncio.wait(
                self.running_tasks.values(), 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for completed_task in done:
                # Find task_id for the completed asyncio Task
                tid = next(k for k, v in self.running_tasks.items() if v == completed_task)
                del self.running_tasks[tid]

        if self.bb.state.status in ["completed", "failed"]:
            logger.info("Loop Terminated. Triggering EO distillation...")
            await self.trigger_eo()

    async def _execute_task(self, task: AtomicTask, running_tasks_map: dict[str, asyncio.Task[None]]) -> None:
        """Isolated execution of a single task with MergeGate integration."""
        async with self.concurrency_limit:
            self.step_counter += 1
            local_step_id = self.step_counter
            logger.info(f"Executing Task {local_step_id}: {task.description}")

            # Prepare causal metadata
            sibling_ids = [tid for tid in running_tasks_map if tid != task.id]

            # 1. Prepare Isolated Context (Snapshot)
            context = await self.bb.get_context(task.context_keys + task.required_keys)
            
            # 2. Synthesize Agent
            primary_capability = task.required_capabilities[0] if task.required_capabilities else "search"
            try:
                sub_agent = self.asynth.synthesize(primary_capability, context)
            except ValueError as e:
                logger.error(f"Synthesizer error: {e}")
                await self._record_failure(task, str(e), local_step_id)
                return

            # 3. Execute in Branch
            prompt = f"Task: {task.description}\nExpected Output: {task.expected_output}"
            output = await self._safe_run(sub_agent, prompt)
            
            if isinstance(output, str):
                result = SubAgentResult(
                    task_id=task.id,
                    status="success",
                    output=output
                )
                logger.info(f"Task {task.id} succeeded.")
                
                # 4. MergeGate: Apply ChangeSet
                # For now, we assume the result is a dict or we just update the task_id_result key
                # To support advanced ChangeSets, sub-agents would need to return structured updates.
                # Here we stick to the basic "result" update for shared memory.
                changeset = {f"{task.id}_result": output}
                await self.bb.apply_changeset(task, changeset)
                
                await self.bb.record_step(TrajectoryStep(
                    step_id=local_step_id, 
                    task=task, 
                    result=result,
                    parent_ids=task.depends_on,
                    sibling_ids=sibling_ids
                ))
            else:
                error_msg = f"Task execution failed: {output}"
                logger.error(error_msg)
                await self._record_failure(task, error_msg, local_step_id)

    async def _record_failure(self, task: AtomicTask, error_msg: str, step_id: int) -> None:
        """Records a task-level failure."""
        result = SubAgentResult(task_id=task.id, status="error", output=error_msg)
        await self.bb.record_step(TrajectoryStep(step_id=step_id, task=task, result=result))

    async def trigger_eo(self) -> None:
        """Hook to trigger the EvolutionaryOptimizer."""
        eo = EvolutionaryOptimizer()
        await eo.process_session(self.bb.state.session_id)
