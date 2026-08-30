"""LLM-based semantic de-duplication for strategic issues"""
import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from adventurer.prompt_library import PromptLibrary
from config import LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from llm_utils import invoke_with_retry


# De-duplication is post-decision bookkeeping running on the cheap model, and
# it happens inside the per-turn budget (config.TURN_BUDGET_SECONDS = 600).
# The global LLM defaults (300s x 5 attempts) would let a single dedup call
# blow through the whole turn budget, so this call site caps them — while still
# honouring a *lower* configured value.
DEDUP_TIMEOUT_SECONDS = min(60, LLM_TIMEOUT_SECONDS)
DEDUP_MAX_RETRIES = min(2, LLM_MAX_RETRIES)


class DeduplicationResult(BaseModel):
    """Result of semantic de-duplication check"""

    is_duplicate: bool
    reason: str


def _as_bool(value) -> bool:
    """Coerce a model-supplied duplicate flag to a real bool.

    Structured output normally hands back a validated bool, but some providers
    return a raw dict where "false" is a non-empty (truthy) string. Reading
    that naively would silently discard a genuine new issue.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "duplicate"}
    return bool(value)


class MemoryDeduplicator:
    """Uses LLM to detect semantically similar strategic issues"""

    def __init__(self, llm: ChatOpenAI):
        """
        Initialize deduplicator with a cheap LLM.

        Args:
            llm: ChatOpenAI instance (should be cheap model like gpt-5-nano)
        """
        self.llm = llm
        self.chain = self._create_chain()
        self.logger = logging.getLogger(__name__)

    def _create_chain(self) -> Runnable:
        """Create LangChain chain for de-duplication checking"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_deduplication_system_prompt()),
            ("human", PromptLibrary.get_deduplication_human_prompt())
        ])

        # Use structured output for consistent parsing
        return prompt | self.llm.with_structured_output(DeduplicationResult)

    def is_duplicate(self, new_issue: str, existing_issues: List[str]) -> tuple[bool, str]:
        """
        Check if new issue is semantically similar to existing ones.

        Never raises. This runs after the turn's command has already executed,
        so a transient LLM error or a malformed response must not end the run
        (GitHub issue #2). On failure it **fails open** — reports "not a
        duplicate" so the issue is stored. The worst case is a redundant issue
        in memory; the alternative was losing the playthrough.

        Args:
            new_issue: The new strategic issue to check
            existing_issues: List of existing issue contents

        Returns:
            Tuple of (is_duplicate: bool, reason: str)
        """
        if not existing_issues:
            return False, "No existing issues to compare against"

        # Format existing issues for prompt
        formatted_existing = "\n".join(
            f"{i+1}. {issue}"
            for i, issue in enumerate(existing_issues)
        )

        # Call LLM with timeout + retry, like every other LLM call in the
        # turn path. The extraction is inside the try as well: structured
        # output can hand back None or a bare dict depending on provider.
        try:
            result = invoke_with_retry(
                self.chain.with_config(run_name="Memory Deduplicator"),
                {
                    "new_issue": new_issue,
                    "existing_issues": formatted_existing
                },
                operation_name="Memory Deduplicator",
                timeout_seconds=DEDUP_TIMEOUT_SECONDS,
                max_retries=DEDUP_MAX_RETRIES,
            )

            if result is None:
                raise ValueError("de-duplication returned no result")

            if isinstance(result, dict):
                is_dup = result.get("is_duplicate")
                reason = result.get("reason", "")
            else:
                is_dup = result.is_duplicate
                reason = getattr(result, "reason", "")

            return _as_bool(is_dup), str(reason)

        except Exception as e:
            # Fail open: store the memory rather than kill the session.
            self.logger.error(
                f"[MemoryDeduplicator] De-duplication failed after "
                f"{DEDUP_MAX_RETRIES} attempt(s); treating '{new_issue}' as NOT "
                f"a duplicate: {e}",
                exc_info=True,
            )
            return False, f"De-duplication unavailable ({e}); stored without semantic check"
