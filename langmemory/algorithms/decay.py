"""
Decay Worker - background importance decay and tier demotion.

Runs as an asyncio task (non-blocking). Processes HOT tier every cycle,
then checks WARM for cold demotion. Never blocks the insert/retrieve hot path.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List

from langmemory.core.node import MemoryNode, MemoryTier
from langmemory.scoring.importance import ImportanceScorer
from langmemory.structures.importance_heap import ImportanceHeap
from langmemory.structures.lsm_tiers import LSMTierManager

logger = logging.getLogger(__name__)


@dataclass
class DecayReport:
    timestamp: float = field(default_factory=time.time)
    nodes_processed: int = 0
    nodes_demoted_warm: int = 0
    nodes_demoted_cold: int = 0
    nodes_pruned: int = 0
    duration_ms: float = 0.0


class DecayWorker:
    """
    Background worker that applies importance decay and tier demotion.

    Usage:
        worker = DecayWorker(tiers, heap, scorer, config)
        task = asyncio.create_task(worker.run())
        # Later:
        worker.stop()
        await task
    """

    def __init__(
        self,
        tiers: LSMTierManager,
        heap: ImportanceHeap,
        scorer: ImportanceScorer,
        interval_seconds: int = 3600,
        hot_threshold: float = 0.3,
        warm_threshold: float = 0.05,
        prune_threshold: float = 0.001,
    ) -> None:
        self._tiers = tiers
        self._heap = heap
        self._scorer = scorer
        self._interval = interval_seconds
        self._hot_threshold = hot_threshold
        self._warm_threshold = warm_threshold
        self._prune_threshold = prune_threshold
        self._running = False
        self._reports: List[DecayReport] = []

    async def run(self) -> None:
        """Run decay cycles indefinitely until stop() is called."""
        self._running = True
        logger.info("DecayWorker started (interval=%ds)", self._interval)
        while self._running:
            await asyncio.sleep(self._interval)
            if self._running:
                report = self.run_once()
                self._reports.append(report)
                logger.debug(
                    "DecayWorker cycle: %d processed, %d->warm, %d->cold, %.1fms",
                    report.nodes_processed,
                    report.nodes_demoted_warm,
                    report.nodes_demoted_cold,
                    report.duration_ms,
                )

    def stop(self) -> None:
        self._running = False

    def run_once(self) -> DecayReport:
        """Execute one decay cycle synchronously. Useful for testing."""
        start = time.time()
        report = DecayReport()
        now = time.time()

        # Process HOT tier
        hot_nodes = list(self._tiers.iter_hot())
        for node in hot_nodes:
            self._scorer.update(node, now=now)
            self._heap.update_priority(node)
            report.nodes_processed += 1

            if node.current_importance < self._prune_threshold:
                self._tiers.demote(node, MemoryTier.COLD)
                self._heap.remove(node.node_id)
                report.nodes_pruned += 1
            elif node.current_importance < self._hot_threshold:
                self._tiers.demote(node, MemoryTier.WARM)
                report.nodes_demoted_warm += 1

        report.duration_ms = (time.time() - start) * 1000
        return report

    @property
    def last_report(self) -> DecayReport | None:
        return self._reports[-1] if self._reports else None
