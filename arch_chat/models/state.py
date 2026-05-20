from pydantic import BaseModel, Field


class DraftCode(BaseModel):
    code: str = Field(description="Python code to solve the user's request.")
    explanation: str = Field(description="Brief explanation of the WHAT and the WHY, while the HOW should be interpreted as the codes itself.")


class Critique(BaseModel):
    has_errors: bool
    is_efficient: bool
    suggested_improvements: list[str]
    critique_summary: str


class RefinedCode(BaseModel):
    refined_code: str
    refinement_summary: str


class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of tool/sub-questions.")


class VerificationResult(BaseModel):
    is_successful: bool = Field(description="True if observation answers the sub-question.")
    reasoning: str


class ControllerDecision(BaseModel):
    next_agent: str = Field(description="Specialist name or FINISH.")
    reasoning: str


class MetacognitiveAnalysis(BaseModel):
    confidence: float = Field(description="0.0~1.0 confidence in safely answering.")
    strategy: str = Field(description="'reason_directly' | 'use_tool' | 'escalate'.")
    reasoning: str
    tool_to_use: str | None = None


class RoutingDecision(BaseModel):
    architecture: str = Field(
        description="Exact architecture name from the allowed list, e.g. meta_controller, reflection."
    )
    confidence: float = Field(
        description="a float num in range of[0.0, 1.0], representing the confidence in this routing choice."
    )
    reasoning: str = Field(
        description="Brief explanation of why this architecture fits the user message."
    )


class EmailDraft(BaseModel):
    subject: str
    body: str


class EmailCritique(BaseModel):
    is_approved: bool
    issues: list[str]
    feedback: str


class Node(BaseModel):
    id: str = Field(description="Unique entity identifier.")
    type: str = Field(description="Entity type, e.g. Person, Company.")


class Relationship(BaseModel):
    source: Node
    target: Node
    type: str = Field(description="Relationship verb in ALL_CAPS.")


class KnowledgeGraphExtract(BaseModel):
    relationships: list[Relationship]


class PuzzleMove(BaseModel):
    move_description: str
    item_to_move: str | None = None


class PuzzleMoves(BaseModel):
    moves: list[PuzzleMove]
