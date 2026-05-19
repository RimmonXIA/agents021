from arch_chat.models.state import RoutingDecision
from arch_chat.router.classifier import has_code_intent, heuristic_route, is_conversational
from arch_chat.router.meta_controller import validate_routing


def test_heuristic_dry_run():
    assert heuristic_route("Please publish this post to Twitter") == "dry_run"


def test_heuristic_episodic():
    assert heuristic_route("Remember that I am allergic to peanuts") == "episodic_memory"


def test_heuristic_tot():
    assert heuristic_route("Solve the wolf goat cabbage river puzzle") == "tot"


def test_heuristic_cellular_automata():
    assert heuristic_route("Run cellular automata pathfinding on a grid") == "cellular_automata"


def test_heuristic_metacognitive():
    assert heuristic_route("What prescription dosage should I take?") == "metacognitive"


def test_heuristic_graph():
    assert heuristic_route("Alice works for Acme Corp") == "graph_memory"


def test_heuristic_reflection():
    assert heuristic_route("Write a python function to sort a list") == "reflection"


def test_heuristic_who_are_you():
    assert heuristic_route("who are you?") == "meta_controller"


def test_heuristic_hello():
    assert heuristic_route("hello") == "meta_controller"


def test_is_conversational_short():
    assert is_conversational("who are you?") is True


def test_has_code_intent_true():
    assert has_code_intent("Write a python function to sort a list") is True


def test_has_code_intent_false():
    assert has_code_intent("who are you?") is False


def test_validate_routing_reflection_override():
    decision = RoutingDecision(
        architecture="reflection",
        confidence=0.9,
        reasoning="LLM picked reflection",
    )
    validated = validate_routing("who are you?", decision)
    assert validated.architecture == "meta_controller"


def test_validate_routing_reflection_allowed_with_code():
    decision = RoutingDecision(
        architecture="reflection",
        confidence=0.9,
        reasoning="Code request",
    )
    validated = validate_routing("Write a python function to sort a list", decision)
    assert validated.architecture == "reflection"


def test_validate_routing_low_confidence():
    decision = RoutingDecision(
        architecture="planning",
        confidence=0.3,
        reasoning="Uncertain",
    )
    validated = validate_routing("tell me something", decision)
    assert validated.architecture == "meta_controller"


def test_validate_routing_specialized_without_keywords():
    decision = RoutingDecision(
        architecture="tot",
        confidence=0.8,
        reasoning="Wrong pick",
    )
    validated = validate_routing("who are you?", decision)
    assert validated.architecture == "meta_controller"
