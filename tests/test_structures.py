"""
Tests for core DSA structures.
Each structure is tested independently - no LLM or embedding dependency.
"""
import time

import pytest

from langmemory.structures.bloom_filter import BloomFilter
from langmemory.structures.concept_trie import ConceptTrie
from langmemory.structures.importance_heap import ImportanceHeap
from langmemory.structures.merkle_tree import MerkleTree
from langmemory.structures.skip_list import TemporalSkipList
from langmemory.core.node import MemoryNode, MemoryType


# ─── Bloom Filter ─────────────────────────────────────────────────────────────

class TestBloomFilter:
    def test_add_and_check(self):
        bf = BloomFilter(capacity=1000)
        key = b"hello world"
        assert not bf.might_contain(key)
        bf.add(key)
        assert bf.might_contain(key)

    def test_no_false_negatives(self):
        """A key that was added must always be found."""
        bf = BloomFilter(capacity=1000)
        keys = [f"key_{i}".encode() for i in range(100)]
        for k in keys:
            bf.add(k)
        for k in keys:
            assert bf.might_contain(k), f"{k} should be in bloom filter"

    def test_false_positive_rate_reasonable(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        for i in range(100):
            bf.add(f"item_{i}".encode())
        fp_rate = bf.false_positive_rate()
        assert fp_rate < 0.1  # Should be well below 10%

    def test_serialization_roundtrip(self):
        bf = BloomFilter(capacity=500)
        bf.add(b"test1")
        bf.add(b"test2")
        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)
        assert bf2.might_contain(b"test1")
        assert bf2.might_contain(b"test2")
        assert not bf2.might_contain(b"test3")

    def test_len(self):
        bf = BloomFilter(capacity=100)
        assert len(bf) == 0
        bf.add(b"a")
        bf.add(b"b")
        assert len(bf) == 2


# ─── Skip List ────────────────────────────────────────────────────────────────

class TestTemporalSkipList:
    def test_insert_and_range(self):
        sl = TemporalSkipList()
        sl.insert(1.0, "a")
        sl.insert(2.0, "b")
        sl.insert(3.0, "c")
        result = sl.range_query(1.0, 2.0)
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_tail(self):
        sl = TemporalSkipList()
        for i in range(10):
            sl.insert(float(i), f"node_{i}")
        tail = sl.tail(3)
        assert "node_9" in tail
        assert "node_8" in tail
        assert "node_7" in tail
        assert "node_0" not in tail

    def test_delete(self):
        sl = TemporalSkipList()
        sl.insert(1.0, "x")
        sl.insert(2.0, "y")
        assert sl.delete(1.0, "x") is True
        result = sl.range_query(0.0, 3.0)
        assert "x" not in result
        assert "y" in result

    def test_duplicate_timestamps(self):
        sl = TemporalSkipList()
        sl.insert(1.0, "a")
        sl.insert(1.0, "b")
        result = sl.range_query(1.0, 1.0)
        assert "a" in result
        assert "b" in result

    def test_len(self):
        sl = TemporalSkipList()
        assert len(sl) == 0
        sl.insert(1.0, "x")
        sl.insert(2.0, "y")
        assert len(sl) == 2
        sl.delete(1.0, "x")
        assert len(sl) == 1


# ─── Merkle Tree ──────────────────────────────────────────────────────────────

class TestMerkleTree:
    def _make_node(self, content: str) -> MemoryNode:
        return MemoryNode(content=content)

    def test_insert_computes_hash(self):
        tree = MerkleTree()
        node = self._make_node("hello")
        tree.insert(node)
        assert node.merkle_hash != b""

    def test_verify_valid_node(self):
        tree = MerkleTree()
        node = self._make_node("test content")
        tree.insert(node)
        assert tree.verify(node.node_id) is True

    def test_tamper_detection(self):
        tree = MerkleTree()
        node = self._make_node("original content")
        tree.insert(node)
        # Tamper with the node's content hash
        node.content_hash = b"\x00" * 32
        assert tree.verify(node.node_id) is False

    def test_parent_child_chain(self):
        tree = MerkleTree()
        parent = self._make_node("parent")
        child = self._make_node("child")
        tree.insert(parent)
        tree.insert(child, parent_id=parent.node_id)
        assert child.parent_id == parent.node_id
        assert child.node_id in parent.children_ids
        assert tree.verify_chain(child.node_id) is True

    def test_root_hash_changes_on_insert(self):
        tree = MerkleTree()
        h1 = tree.root_hash()
        tree.insert(self._make_node("a"))
        h2 = tree.root_hash()
        tree.insert(self._make_node("b"))
        h3 = tree.root_hash()
        assert h1 != h2
        assert h2 != h3

    def test_get_proof(self):
        tree = MerkleTree()
        parent = self._make_node("root")
        child = self._make_node("leaf")
        tree.insert(parent)
        tree.insert(child, parent_id=parent.node_id)
        proof = tree.get_proof(child.node_id)
        assert len(proof) == 2  # child + parent


# ─── Importance Heap ──────────────────────────────────────────────────────────

class TestImportanceHeap:
    def _make_node(self, importance: float, nid: str = None) -> MemoryNode:
        n = MemoryNode(content=f"node-{importance}")
        n.current_importance = importance
        if nid:
            n.node_id = nid
        return n

    def test_push_and_extract_top_k(self):
        heap = ImportanceHeap()
        for i, imp in enumerate([0.1, 0.9, 0.5, 0.7, 0.3]):
            n = self._make_node(imp, nid=f"n{i}")
            heap.push(n)
        top2 = heap.extract_top_k(2)
        importances = [n.current_importance for n in top2]
        assert importances[0] >= importances[1]
        assert importances[0] == 0.9

    def test_extract_min(self):
        heap = ImportanceHeap()
        heap.push(self._make_node(0.8, "a"))
        heap.push(self._make_node(0.2, "b"))
        heap.push(self._make_node(0.5, "c"))
        min_node = heap.extract_min()
        assert min_node is not None
        assert min_node.current_importance == 0.2

    def test_update_priority(self):
        heap = ImportanceHeap()
        node = self._make_node(0.1, "x")
        heap.push(node)
        node.current_importance = 0.9
        heap.update_priority(node)
        top = heap.extract_top_k(1)
        assert top[0].current_importance == 0.9

    def test_remove(self):
        heap = ImportanceHeap()
        node = self._make_node(0.5, "del_me")
        heap.push(node)
        assert "del_me" in heap
        heap.remove("del_me")
        assert "del_me" not in heap

    def test_len(self):
        heap = ImportanceHeap()
        assert len(heap) == 0
        heap.push(self._make_node(0.5, "a"))
        heap.push(self._make_node(0.6, "b"))
        assert len(heap) == 2


# ─── Concept Trie ─────────────────────────────────────────────────────────────

class TestConceptTrie:
    def test_insert_and_exact_search(self):
        trie = ConceptTrie()
        trie.insert(["Python/types"], "node1")
        result = trie.exact_search("Python/types")
        assert "node1" in result

    def test_prefix_search_includes_children(self):
        trie = ConceptTrie()
        trie.insert(["Python/types/generics"], "node1")
        trie.insert(["Python/types/literals"], "node2")
        trie.insert(["Python/async"], "node3")
        trie.insert(["JavaScript"], "node4")

        result = trie.prefix_search("Python")
        assert "node1" in result
        assert "node2" in result
        assert "node3" in result
        assert "node4" not in result

    def test_subtree_size(self):
        trie = ConceptTrie()
        trie.insert(["ML/optimization/SGD"], "n1")
        trie.insert(["ML/optimization/Adam"], "n2")
        trie.insert(["ML/training"], "n3")
        assert trie.subtree_size("ML/optimization") == 2
        assert trie.subtree_size("ML") == 3

    def test_remove(self):
        trie = ConceptTrie()
        trie.insert(["Python/types"], "node1")
        trie.remove(["Python/types"], "node1")
        result = trie.exact_search("Python/types")
        assert "node1" not in result

    def test_multiple_concepts_per_node(self):
        trie = ConceptTrie()
        trie.insert(["Python", "programming"], "multi_node")
        assert "multi_node" in trie.prefix_search("Python")
        assert "multi_node" in trie.prefix_search("programming")


# ─── Integration: full insert + retrieve ──────────────────────────────────────

class TestIntegration:
    def test_basic_insert_retrieve(self):
        from langmemory import LangMemory
        mc = LangMemory()
        mc.insert("User prefers concise responses")
        mc.insert("User is building a Python web service")
        mc.insert("The project uses FastAPI and PostgreSQL")

        results = mc.retrieve("What framework is being used?", top_k=3)
        assert len(results) > 0
        contents = [r.node.content for r in results]
        # At least one result should be about the project
        assert any("FastAPI" in c or "Python" in c or "project" in c.lower() for c in contents)

    def test_deduplication(self):
        from langmemory import LangMemory
        mc = LangMemory()
        n1 = mc.insert("User prefers Python")
        n2 = mc.insert("User prefers Python")  # exact duplicate
        # Should return same node (deduplication)
        assert n1.node_id == n2.node_id

    def test_integrity_verification(self):
        from langmemory import LangMemory
        mc = LangMemory()
        mc.insert("Memory 1")
        mc.insert("Memory 2")
        report = mc.verify_integrity()
        assert report.is_valid is True
        assert report.total_nodes == 2

    def test_stats(self):
        from langmemory import LangMemory
        mc = LangMemory()
        mc.insert("Test memory")
        stats = mc.stats()
        assert stats["hot_count"] == 1
        assert stats["total_indexed"] == 1

    def test_context_window(self):
        from langmemory import LangMemory
        mc = LangMemory()
        mc.insert("User's name is Alice")
        mc.insert("Alice works at Anthropic")
        context = mc.get_context_window("Who is the user?", token_budget=500)
        assert "[MEMORIES]" in context
        assert "[/MEMORIES]" in context

    def test_decay_runs(self):
        from langmemory import LangMemory
        mc = LangMemory()
        mc.insert("Temporary working memory", memory_type=MemoryType.WORKING)
        mc.run_decay()
        stats = mc.stats()
        assert stats is not None  # Just verify it doesn't crash
