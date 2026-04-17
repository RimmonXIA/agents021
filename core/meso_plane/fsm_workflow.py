# core/meso_plane/fsm_workflow.py

import json
from typing import Dict, Any, List
from agno.workflow import Workflow
from dataclasses import dataclass

@dataclass
class RunResponse:
    status: str
    message: str = ""
    content: str = ""
    session_id: str = ""
from agno.db.sqlite import SqliteDb

from core.macro_plane.intent_network import IntentNetworkAgent
from core.micro_plane.capability_registry import (
    get_search_capability,
    get_data_extraction_capability,
    get_code_analysis_capability
)

class MacroMesoMicroWorkflow(Workflow):
    def __init__(self, session_id: str):
        super().__init__(
            name="OrchestrationBus",
            session_id=session_id,
            db=SqliteDb(
                session_table="meso_plane_state",
                db_url="sqlite:///state/workflow_storage.db"
            )
        )
        self.macro_agent = IntentNetworkAgent()
        self.session_state: Dict[str, Any] = {"telemetry_events": []}
        
        self.capabilities = {
            "search": get_search_capability,
            "extract": get_data_extraction_capability,
            "analyze_code": get_code_analysis_capability
        }

    def run(self, user_objective: str) -> RunResponse:
        self.add_telemetry_event("WORKFLOW_START", {"objective": user_objective})

        try:
            plan_response = self.macro_agent.run(user_objective)
            content = plan_response.content if hasattr(plan_response, 'content') else str(plan_response)
            
            # Remove any possible markdown wrappers
            if isinstance(content, str):
                content = content.replace("```json", "").replace("```", "").strip()

            execution_plan = json.loads(content)
            tasks: List[Dict[str, Any]] = execution_plan.get("tasks", [])
        except json.JSONDecodeError as e:
            self.add_telemetry_event("MACRO_ERROR", {"error": "Invalid JSON plan generated", "details": str(e)})
            return RunResponse(status="FAILED", message="Macro-plane failed to produce a valid deterministic plan.")
        except Exception as e:
            self.add_telemetry_event("MACRO_ERROR", {"error": "Macro-plane execution failed", "details": str(e)})
            return RunResponse(status="FAILED", message="Macro-plane execution error.")

        self.add_telemetry_event("STATE_TRANSITION", {"state": "PLAN_GENERATED", "tasks_count": len(tasks)})

        results = []
        for task in tasks:
            capability_name = task.get("capability")
            task_input = task.get("input")

            self.add_telemetry_event("STATE_TRANSITION", {"state": "TASK_START", "capability": capability_name})

            if capability_name not in self.capabilities:
                self.add_telemetry_event("ROUTING_ERROR", {"error": "Unknown capability", "capability": capability_name})
                return RunResponse(status="FAILED", message=f"Pipeline halted: Capability '{capability_name}' not registered.")

            capability_factory = self.capabilities[capability_name]
            ephemeral_agent = capability_factory()

            try:
                task_response = ephemeral_agent.run(task_input)
                result_payload = task_response.content if hasattr(task_response, 'content') else str(task_response)
                results.append({"capability": capability_name, "result": result_payload})
                
                self.add_telemetry_event("STATE_TRANSITION", {"state": "TASK_COMPLETE", "capability": capability_name})
                
            except Exception as e:
                self.add_telemetry_event("MICRO_ERROR", {"error": "Task execution failed", "capability": capability_name, "details": str(e)})
                return RunResponse(status="FAILED", message=f"Pipeline halted: Micro-plane failure in '{capability_name}'.")

        self.add_telemetry_event("STATE_TRANSITION", {"state": "PIPELINE_COMPLETE"})
        
        return RunResponse(
            status="SUCCESS",
            content=json.dumps({"final_results": results}),
            session_id=self.session_id
        )

    def add_telemetry_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event_record = {
            "event_type": event_type,
            "payload": payload
        }
        self.session_state["telemetry_events"].append(event_record)
        
        # If db is attached, we can update the session
        try:
            self.update_session_state({"telemetry_events": self.session_state["telemetry_events"]})
        except Exception as e:
            pass # Session might not be initialized yet
