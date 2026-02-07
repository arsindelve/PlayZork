"""LLM-based semantic de-duplication for strategic issues"""
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from adventurer.prompt_library import PromptLibrary


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
            ("system", PromptLibrary.get_deduplication_system_prompt()),
            ("human", PromptLibrary.get_deduplication_human_prompt())
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
