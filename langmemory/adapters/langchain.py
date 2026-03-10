"""
LangChain adapter for LangMemory.

Drop-in replacement for any LangChain BaseChatMemory.

Usage:
    from langmemory import LangMemory
    from langmemory.adapters.langchain import LangChainMemory
    from langchain.chains import ConversationChain

    mc = LangMemory()
    memory = LangChainMemory(chain=mc)
    conv = ConversationChain(llm=llm, memory=memory)
"""
from __future__ import annotations

from typing import Any, Dict, List

from langmemory.core.chain import LangMemory
from langmemory.core.node import MemoryType


class LangMemoryMemory:
    """
    LangChain-compatible memory backed by LangMemory.

    Implements the BaseChatMemory interface without requiring langchain
    as a hard dependency (duck-typing compatible).
    """

    memory_key: str = "history"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    return_messages: bool = False

    def __init__(
        self,
        chain: LangMemory,
        session_id: str = "",
        context_token_budget: int = 2000,
    ) -> None:
        self.chain = chain
        self.session_id = session_id
        self.context_token_budget = context_token_budget
        self._chat_history: List[Dict[str, str]] = []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Called by LangChain after each chain invocation.
        Saves human input + AI output as memories.
        """
        human_input = inputs.get("input", inputs.get("human_input", ""))
        ai_output = outputs.get("output", outputs.get("response", ""))

        if human_input:
            self.chain.insert(
                content=f"{self.human_prefix}: {human_input}",
                memory_type=MemoryType.EPISODIC,
                source_session_id=self.session_id,
                tags={"role": "human", "session": self.session_id},
            )

        if ai_output:
            self.chain.insert(
                content=f"{self.ai_prefix}: {ai_output}",
                memory_type=MemoryType.EPISODIC,
                source_session_id=self.session_id,
                tags={"role": "ai", "session": self.session_id},
            )

        self._chat_history.append({"human": human_input, "ai": ai_output})

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called by LangChain before each chain invocation.
        Returns relevant memories formatted as context.
        """
        query = inputs.get("input", inputs.get("human_input", ""))
        context = self.chain.get_context_window(
            query=query,
            token_budget=self.context_token_budget,
        )

        if self.return_messages:
            # Return as list of message dicts (for chat models)
            return {self.memory_key: [{"role": "system", "content": context}]}

        return {self.memory_key: context}

    def clear(self) -> None:
        """Clear in-memory chat history (does not wipe LangMemory)."""
        self._chat_history.clear()

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
