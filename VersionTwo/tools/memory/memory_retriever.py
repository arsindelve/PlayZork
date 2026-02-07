from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from .memory_state import Memory
from adventurer.prompt_library import PromptLibrary


class MemoryRetriever:
    """Uses a cheap LLM to intelligently search and summarize memories"""

    def __init__(self, llm: ChatOpenAI):
        """
        Args:
            llm: A cheap LLM for semantic search (e.g., GPT-3.5-turbo)
        """
        self.llm = llm

    def query_memories(
        self,
        query: str,
        memories: List[Memory],
        max_results: int = 5
    ) -> str:
        """
        Semantic search: find memories relevant to a query.

        Args:
            query: Natural language question (e.g., "Where did I see a key?")
            memories: List of all memories to search
            max_results: Max memories to include in response

        Returns:
            Formatted string with relevant memories and answer
        """
        if not memories:
            return "No memories recorded yet."

        # Format all memories for the LLM
        memories_text = "\n".join([
            f"{i+1}. [Turn {m.turn_number} @ {m.location}, importance={m.importance}] {m.content}"
            for i, m in enumerate(memories)
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_memory_query_system_prompt()),
            ("human", PromptLibrary.get_memory_query_human_prompt())
        ])

        response = self.llm.invoke(
            prompt.format_messages(memories=memories_text, query=query)
        )

        return response.content

    def summarize_location_memories(
        self,
        location: str,
        memories: List[Memory]
    ) -> str:
        """
        Summarize all memories about a specific location.

        Args:
            location: Location name (e.g., "West Of House")
            memories: Memories filtered to this location

        Returns:
            Summary of what we learned at this location
        """
        if not memories:
            return f"No memories recorded for {location}."

        memories_text = "\n".join([
            f"- Turn {m.turn_number}: {m.content} (importance: {m.importance})"
            for m in memories
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_memory_location_summary_system_prompt()),
            ("human", PromptLibrary.get_memory_location_summary_human_prompt())
        ])

        response = self.llm.invoke(
            prompt.format_messages(location=location, memories=memories_text)
        )

        return response.content

    def get_top_insights(
        self,
        memories: List[Memory],
        limit: int = 10
    ) -> str:
        """
        Format the top N most important memories.

        Args:
            memories: Top memories (already sorted)
            limit: How many to include

        Returns:
            Formatted string of top memories
        """
        if not memories:
            return "No memories recorded yet."

        result = f"Top {len(memories[:limit])} Most Important Memories:\n\n"

        for i, m in enumerate(memories[:limit], 1):
            result += f"{i}. [Turn {m.turn_number} @ {m.location}] {m.content}\n"
            result += f"   Importance: {m.importance}/1000\n\n"

        return result.strip()
