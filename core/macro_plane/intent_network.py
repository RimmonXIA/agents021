# core/macro_plane/intent_network.py
from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from agno.models.deepseek import DeepSeek

class IntentNetworkAgent(Agent):
    def __init__(self):
        super().__init__(
            model=DeepSeek(),
            name="IntentNetwork",
            description="Decomposes user objectives into strict execution plans.",
            instructions=[
                "Analyze the user objective.",
                "Decompose into a linear sequence of required atomic capabilities.",
                "Allowed capabilities: 'search', 'extract', 'analyze_code'.",
                "Output strictly valid JSON with the schema: {'tasks': [{'capability': 'str', 'input': 'str'}]}",
                "Do not include conversational filler or markdown code blocks."
            ],
        )
