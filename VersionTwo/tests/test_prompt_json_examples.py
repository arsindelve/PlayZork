"""JSON examples in prompts must be valid JSON as the model sees them (#28).

`prompt_library` mixes two consumption paths and they need opposite escaping:

  * TEMPLATE prompts go through ChatPromptTemplate, which collapses `{{` -> `{`.
  * PLAIN-STRING prompts are f-strings handed straight to the model, where the
    f-string itself collapses `{{` -> `{` exactly once.

Writing the template convention (`{{{{`) in a plain-string prompt collapses
only once and shows the model `{{` — teaching malformed JSON to a 14B model
whose entire job depends on emitting well-formed structured output.

The issue reported this as affecting every JSON example. It did not: the
template-rendered prompts were already correct. Only the plain-string path
was wrong, which is exactly why a test that renders each prompt *the way it is
actually consumed* is worth more than a source-text grep.
"""
import json
import re

import pytest
from langchain_core.prompts import ChatPromptTemplate

from adventurer.prompt_library import PromptLibrary


def json_blocks(text):
    """Brace-balanced top-level blocks, so nested objects are not split."""
    blocks, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:i + 1])
    return blocks


# Prompts handed to the model as plain strings (no template rendering).
PLAIN_STRING_PROMPTS = {
    "issue_closed_analysis": lambda: PromptLibrary.get_issue_closed_analysis_prompt(
        "- [ID:1, Importance:500/1000] open the mailbox", "history", "West Of House", "Opened."
    ),
    "observer_observation": lambda: PromptLibrary.get_observer_observation_prompt(
        "You see a mailbox.", "West Of House", "history", "tracked issues"
    ),
    "death_detection_human": lambda: PromptLibrary.get_death_detection_human_prompt(),
    "inventory_analyzer_system": lambda: PromptLibrary.get_inventory_analyzer_system_prompt(),
}


@pytest.mark.parametrize("name", sorted(PLAIN_STRING_PROMPTS))
def test_plain_string_prompt_examples_are_valid_json(name):
    text = PLAIN_STRING_PROMPTS[name]()

    for block in json_blocks(text):
        if '"' not in block:
            continue  # prose braces, not a JSON example
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            pytest.fail(f"{name}: model is shown invalid JSON: {block!r} ({e})")


@pytest.mark.parametrize("name", sorted(PLAIN_STRING_PROMPTS))
def test_plain_string_prompts_never_show_doubled_braces(name):
    text = PLAIN_STRING_PROMPTS[name]()

    assert "{{" not in text, f"{name}: over-escaped — the model sees literal '{{{{'"
    assert "}}" not in text


def test_template_prompt_examples_are_valid_json_after_rendering():
    """The other path: these are correct precisely because they ARE escaped."""
    rendered = ChatPromptTemplate.from_messages([
        ("system", PromptLibrary.get_decision_agent_evaluation_prompt()),
        ("human", PromptLibrary.get_decision_agent_human_prompt()),
    ]).format_messages(
        locationName="West Of House", score=0, moves=1, game_response="r",
        research_context="c", agent_proposals="p",
    )

    for message in rendered:
        for block in json_blocks(message.content):
            if '"' in block:
                json.loads(block)


def test_the_issue_closed_prompt_still_uses_unechoable_example_ids():
    """Guards #19 while #28 rewrites the same block: the placeholder IDs must
    stay far above any real rowid, or an example-echoing model closes real
    issues."""
    text = PromptLibrary.get_issue_closed_analysis_prompt("(none)", "h", "X", "r")

    ids = [int(n) for block in re.findall(r'"closed_issue_ids"\s*:\s*\[([^\]]*)\]', text)
           for n in re.findall(r"\d+", block)]
    ids += [int(n) for n in re.findall(r"\[ID:(\d+)", text)]

    assert ids
    assert all(i >= 9000 for i in ids), f"echoable real-looking IDs in prompt: {ids}"
