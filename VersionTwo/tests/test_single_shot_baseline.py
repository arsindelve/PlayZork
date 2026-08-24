"""The experiment's control arm (thesis).

The research question is whether multi-agent deliberation lets a weak model
solve what it cannot solve by direct inference. That needs a control: the SAME
model, given the SAME information, asked once.

These tests pin the fairness properties. A weak baseline would make the whole
comparison meaningless, so the control must be demonstrably well-informed —
the independent variable is the deliberation, not the context.
"""
import asyncio
import importlib
import os
import re
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

import llm_utils
from adventurer.adventurer_response import AdventurerResponse
from adventurer.prompt_library import PromptLibrary
from adventurer.single_shot_service import SingleShotService


def _toolkits():
    history = SimpleNamespace(state=SimpleNamespace(
        get_full_summary=lambda: "recent summary text",
        get_long_running_summary=lambda: "the story so far",
        get_recent_turns=lambda n: [
            SimpleNamespace(turn_number=1, player_command="LOOK",
                            game_response="West Of House", location="West Of House",
                            score=0, moves=1),
            SimpleNamespace(turn_number=2, player_command="OPEN MAILBOX",
                            game_response="A leaflet.", location="West Of House",
                            score=0, moves=2),
        ]))
    mapper = SimpleNamespace(state=SimpleNamespace(
        get_exits_from=lambda loc: [("NORTH", "North of House")],
        get_all_transitions=lambda: [
            SimpleNamespace(from_location="West Of House", direction="NORTH",
                            to_location="North of House")]))
    inventory = SimpleNamespace(state=SimpleNamespace(get_items=lambda: ["leaflet"]))
    memory = SimpleNamespace(state=SimpleNamespace(get_top_memories=lambda **k: [
        SimpleNamespace(content="open the trap door", importance=700)]))
    return history, memory, mapper, inventory


def _run(monkeypatch, response=None):
    captured = {}

    async def fake(chain, payload, operation_name="", **kw):
        captured["inputs"] = payload
        captured["operation"] = operation_name
        return response or AdventurerResponse(
            command="NORTH", reason="explore", moved="NORTH")

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake)
    monkeypatch.setattr("adventurer.single_shot_service.get_expensive_llm",
                        lambda temperature=0: SimpleNamespace(
                            with_structured_output=lambda s: RunnableLambda(lambda _: None)))
    from game_logger import GameLogger
    GameLogger.get_instance("test-single-shot")

    service = SingleShotService(*_toolkits())
    result = asyncio.run(service.handle_user_input(
        SimpleNamespace(LocationName="West Of House", Response="You are here.",
                        Score=0, Moves=2, Inventory=["leaflet"]),
        turn_number=3, player_command="OPEN MAILBOX"))
    return captured, result


def test_it_makes_exactly_one_llm_call(monkeypatch):
    """The whole point of the control arm."""
    calls = []

    async def counting(chain, payload, operation_name="", **kw):
        calls.append(operation_name)
        return AdventurerResponse(command="NORTH", reason="r", moved="NORTH")

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", counting)
    monkeypatch.setattr("adventurer.single_shot_service.get_expensive_llm",
                        lambda temperature=0: SimpleNamespace(
                            with_structured_output=lambda s: RunnableLambda(lambda _: None)))
    from game_logger import GameLogger
    GameLogger.get_instance("test-single-shot")

    asyncio.run(SingleShotService(*_toolkits()).handle_user_input(
        SimpleNamespace(LocationName="X", Response="r", Score=0, Moves=1, Inventory=[]),
        turn_number=1, player_command="LOOK"))

    assert len(calls) == 1


@pytest.mark.parametrize("field", [
    "inventory", "exits", "already_tried", "tracked_issues",
    "known_map", "recent_turns", "full_summary", "long_summary",
])
def test_the_control_receives_everything_the_treatment_assembles(monkeypatch, field):
    """Fairness: the independent variable is the DELIBERATION, not the context.
    A control starved of information would make the comparison meaningless."""
    captured, _ = _run(monkeypatch)

    assert field in captured["inputs"]
    assert captured["inputs"][field], f"{field} was empty — baseline is under-informed"


def test_it_returns_the_same_interface_as_the_multi_agent_arm(monkeypatch):
    """GameSession must be able to run either arm without knowing which."""
    _, result = _run(monkeypatch)

    assert len(result) == 10
    decision, agents, explorer, loop, interaction, closed, observer, prompt, rtc, dtc = result
    assert isinstance(decision, AdventurerResponse)
    assert agents == [] and explorer is None and interaction is None
    assert isinstance(prompt, str) and prompt


def test_it_uses_the_same_model_tier_as_the_arbiter():
    """Comparing architectures requires holding the model constant."""
    import inspect

    source = inspect.getsource(SingleShotService.__init__)

    assert "get_expensive_llm" in source
    assert "get_cheap_llm" not in source


def test_the_prompt_forbids_repeating_dead_commands():
    """The control gets #18's suppression list too — otherwise it would
    deadlock the way the treatment did, and the comparison would measure that
    instead of the architecture."""
    system = PromptLibrary.get_single_shot_system_prompt()

    assert "ALREADY TRIED HERE" in system
    assert "deterministic" in system.lower()


def test_prompt_variables_match_what_the_service_supplies(monkeypatch):
    """A missing key raises at render time, mid-run."""
    captured, _ = _run(monkeypatch)

    declared = set(re.findall(r"\{(\w+)\}", PromptLibrary.get_single_shot_human_prompt()))
    assert declared == set(captured["inputs"]), declared ^ set(captured["inputs"])


def test_the_condition_is_a_runtime_setting():
    import config

    try:
        os.environ["PLAYZORK_CONDITION"] = "single_shot"
        cfg = importlib.reload(config)
        assert cfg.EXPERIMENT_CONDITION == "single_shot"

        os.environ["PLAYZORK_CONDITION"] = "nonsense"
        with pytest.raises(ValueError, match="PLAYZORK_CONDITION"):
            importlib.reload(config)
    finally:
        os.environ["PLAYZORK_CONDITION"] = "multi_agent"
        importlib.reload(config)
