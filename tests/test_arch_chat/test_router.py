from arch_chat.models.state import RoutingDecision
from arch_chat.registry import ARCH_REGISTRY
from arch_chat.router.classifier import has_code_intent, is_conversational
from arch_chat.router.meta_controller import normalize_decision
from arch_chat.router.prompts import (
    ARCH_ROUTING_HINTS,
    build_architecture_catalog,
    build_router_system_prompt,
    build_router_user_prompt,
)


def test_build_router_prompt_contains_all_architectures():
    prompt = build_router_system_prompt()
    for name in ARCH_REGISTRY:
        assert name in prompt
    assert "meta_controller" in prompt
    assert "reflection" in prompt


def test_architecture_catalog_lists_17():
    catalog = build_architecture_catalog()
    assert catalog.count("\n") >= 16  # 17 entries, 16 newlines


def test_routing_hints_cover_all_archs():
    assert len(ARCH_ROUTING_HINTS) == len(ARCH_REGISTRY)


def test_normalize_valid_name():
    decision = RoutingDecision(
        architecture="reflection",
        confidence=0.9,
        reasoning="code task",
    )
    normalized = normalize_decision(decision)
    assert normalized.architecture == "reflection"
    assert normalized.confidence == 0.9


def test_normalize_unknown_falls_back_to_meta():
    decision = RoutingDecision(
        architecture="nonexistent_arch",
        confidence=0.9,
        reasoning="bad pick",
    )
    normalized = normalize_decision(decision)
    assert normalized.architecture == "meta_controller"


def test_normalize_hyphenated_name():
    decision = RoutingDecision(
        architecture="meta-controller",
        confidence=0.8,
        reasoning="general chat",
    )
    normalized = normalize_decision(decision)
    assert normalized.architecture == "meta_controller"


def test_is_conversational_short():
    assert is_conversational("who are you?") is True


def test_has_code_intent_true():
    assert has_code_intent("Write a python function to sort a list") is True


def test_has_code_intent_false():
    assert has_code_intent("who are you?") is False


def test_build_user_prompt_includes_message():
    msg = "Compare AAPL and MSFT"
    prompt = build_router_user_prompt(msg)
    assert msg in prompt
