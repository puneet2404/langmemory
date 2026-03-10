"""
LangMemory — Memory that behaves like a system, not a log.

AGE Memory turns recall into a tiered runtime built from queues, indexes,
graphs, and promotion logic. Fast. Inspectable. Durable.

Quick start:
    from langmemory import LangMemory

    mc = LangMemory()
    mc.insert("User prefers Python over JavaScript")
    results = mc.retrieve("What language does the user like?")
    print(results[0].node.content)
"""
from langmemory.core.chain import LangMemory
from langmemory.core.config import ChainConfig
from langmemory.core.node import MemoryNode, MemoryTier, MemoryType

__version__ = "0.1.1"
__all__ = ["LangMemory", "ChainConfig", "MemoryNode", "MemoryTier", "MemoryType"]
