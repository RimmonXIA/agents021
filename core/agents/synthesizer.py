import importlib
from typing import Any

from agno.agent import Agent

from core.utils.logging import get_logger

logger = get_logger(__name__)

# All known capability → template module suffixes
_KNOWN_CAPABILITIES = ["search", "python_execution", "file_analyzer", "planner"]


class AgentSynthesizer:
    """
    (AS) - The Agent Synthesizer.
    Responsible for instantiating the right Agent template based on required capabilities.

    At construction time, eagerly probes all known capability templates so that broken
    transitive imports (e.g. a missing optional dependency) surface as startup errors
    rather than silent runtime failures mid-execution.
    """
    def __init__(self, templates_package: str = "core.agents.templates"):
        self.templates_package = templates_package
        self.available_capabilities: set[str] = set()
        self.broken_capabilities: dict[str, str] = {}
        self._probe_capabilities()

    def _probe_capabilities(self) -> None:
        """
        Eagerly import every known template to distinguish 'missing template file'
        from 'template exists but has a broken transitive dependency'.
        Populates self.available_capabilities and self.broken_capabilities.
        """
        for cap in _KNOWN_CAPABILITIES:
            module_name = f"{self.templates_package}.{cap}_agent"
            try:
                importlib.import_module(module_name)
                self.available_capabilities.add(cap)
                logger.debug(f"Capability probe OK: {cap}")
            except ImportError as e:
                # Exact match → the template file itself is missing
                if e.name == module_name:
                    reason = f"Template file not found: {module_name}"
                else:
                    # Substring / transitive import failure — the template exists
                    # but one of its dependencies is broken.
                    reason = f"Broken dependency in '{cap}' template: {e.name} — {e}"
                self.broken_capabilities[cap] = reason
                logger.warning(f"Capability probe FAILED: {cap} — {reason}")
            except Exception as e:
                self.broken_capabilities[cap] = f"Unexpected error: {e}"
                logger.warning(f"Capability probe ERROR: {cap} — {e}")

    def synthesize(self, capability: str, context: dict[str, Any]) -> Agent:
        """
        Dynamically loads an agent template based on the capability string.
        Injects context into the agent's instructions.
        """
        # Fast-fail if we already know this capability is broken
        if capability in self.broken_capabilities:
            raise ValueError(
                f"Capability '{capability}' is unavailable: "
                f"{self.broken_capabilities[capability]}"
            )

        module_name = f"{self.templates_package}.{capability}_agent"
        logger.debug(f"Synthesizing agent for capability: {capability}")

        try:
            module = importlib.import_module(module_name)

            if not hasattr(module, "get_agent"):
                raise ValueError(
                    f"Template module {module_name} must implement "
                    "'get_agent(context: Dict[str, Any]) -> Agent'"
                )

            agent: Agent = module.get_agent(context)
            logger.info(f"Agent synthesized: {agent.name} (Capability: {capability})")
            return agent

        except ImportError as e:
            # Exact match on e.name → the template module itself is absent
            if e.name == module_name:
                error_msg = f"Capability '{capability}' is not supported. Missing template: {module_name}"
            else:
                error_msg = f"Broken dependency in template '{capability}': [{e.name}] {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error during synthesis of '{capability}': {e}")
            raise
