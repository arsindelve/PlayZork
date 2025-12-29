"""LLM-based semantic de-duplication for strategic issues"""
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from config import GAME_NAME


class DeduplicationResult(BaseModel):
    """Result of semantic de-duplication check"""

    is_duplicate: bool
    reason: str


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

    def _create_chain(self) -> Runnable:
        """Create LangChain chain for de-duplication checking"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a de-duplication assistant for a {GAME_NAME} game-playing AI.

Your task: Determine if a NEW strategic issue is semantically similar to EXISTING issues.

Strategic issues are:
- UNSOLVED PUZZLES (e.g., "locked grating blocks path east")
- OBVIOUS THINGS TO TRY (e.g., "get inside the white house")
- MAJOR OBSTACLES (e.g., "troll demands payment to pass")

Rules for determining duplicates:
1. Same core problem = DUPLICATE
   - "Need to get into the white house" ≈ "Find way to enter white house" → DUPLICATE
   - "Locked grating blocks east path" ≈ "Can't go east due to locked grate" → DUPLICATE

2. Different specific details = NOT DUPLICATE
   - "Troll blocks bridge" vs "Cyclops blocks passage" → NOT DUPLICATE
   - "Need light for dark room" vs "Need to cross river" → NOT DUPLICATE

3. Progress updates = DUPLICATE
   - If new issue is just a rephrasing or update of existing issue → DUPLICATE

4. More specific version = DUPLICATE
   - "Need to explore the house" (existing) vs "Check kitchen in house" (new) → DUPLICATE

5. Completely different problem = NOT DUPLICATE
   - Even if same location or similar phrasing

Return your decision as structured output."""),
            ("human", """NEW ISSUE:
{new_issue}

EXISTING ISSUES:
{existing_issues}

Is the NEW issue a duplicate of any existing issue?""")
        ])

        # Use structured output for consistent parsing
        return prompt | self.llm.with_structured_output(DeduplicationResult)

    def is_duplicate(self, new_issue: str, existing_issues: List[str]) -> tuple[bool, str]:
        """
        Check if new issue is semantically similar to existing ones.

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

        # Call LLM
        result = self.chain.invoke({
            "new_issue": new_issue,
            "existing_issues": formatted_existing
        })

        return result.is_duplicate, result.reason
