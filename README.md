# langmemory

**Memory that behaves like a system, not a log.**

AGE Memory turns recall into a tiered runtime built from queues, indexes, graphs, and promotion logic. Fast. Inspectable. Durable.

---

Most LLM memory is a list. You append to it. You search it. It grows until it breaks.

LangMemory is different. Every memory is a node in a living system — assigned an importance, placed in a tier, indexed by time and concept, chained cryptographically to its history, and promoted or demoted automatically as the conversation evolves.

Zero dependencies. Pure Python.

---

## Install

```bash
pip install langmemory
```

No required dependencies. Works out of the box with Python 3.10+.

---

## Quick start

```python
from langmemory import LangMemory

memory = LangMemory()

memory.insert("User is building a Python API with FastAPI")
memory.insert("They prefer minimal dependencies")
memory.insert("The project deploys to AWS Lambda")

results = memory.retrieve("What should I know about the user's stack?")

for r in results:
    print(r.node.content)
    print(f"  score={r.score:.3f}  importance={r.node.current_importance:.3f}")
```

Inject into any LLM prompt:

```python
context = memory.get_context_window(
    query="What does the user care about?",
    token_budget=1000
)

# Pass context directly into your system prompt
response = llm.chat(system=context, user=user_message)
```

---

## How it works

LangMemory is not a vector database wrapper. It is a memory runtime built from six data structures, each solving a specific problem that flat storage cannot.

### The six primitives

| Structure | Complexity | Role |
|---|---|---|
| **Bloom Filter** | O(1) | Deduplication — never store the same memory twice |
| **Skip List** | O(log n) | Temporal index — range queries over time |
| **Merkle Tree** | O(log n) | Integrity chain — every memory is cryptographically linked |
| **LSM Tiers** | O(1) amortized | Hot / Warm / Cold storage — memory cost scales with access patterns |
| **Min-Heap** | O(log n) | Importance-weighted retrieval — surfaces what matters, not just what's recent |
| **Concept Trie** | O(key length) | Hierarchical concept indexing — `Python/types/generics` is a path, not a tag |

### The write path

```
insert(content)
  │
  ├─ Bloom Filter      → duplicate? return existing node
  ├─ Embed             → encode to vector (pluggable backend)
  ├─ MemoryNode        → assign importance, decay rate, concepts
  ├─ Merkle Tree       → compute hash, link to chain
  ├─ Skip List         → register timestamp
  ├─ Concept Trie      → index under concept paths
  ├─ Importance Heap   → push with priority
  └─ HOT tier          → store in memory, compact if over limit
```

### The read path

```
retrieve(query)
  │
  ├─ Path A: Semantic  → importance heap × embedding similarity
  ├─ Path B: Temporal  → skip list range or recent tail
  ├─ Path C: Concept   → trie prefix lookup
  │
  ├─ Score fusion      → (0.5 × semantic) + (0.3 × importance) + (0.2 × recency)
  └─ Top-K             → sorted, access metadata updated
```

### Tiers

Memory moves automatically between tiers as importance decays:

```
HOT   →  RAM          Fast read/write. Recent and frequently accessed nodes.
WARM  →  Disk         Persistent. Moderate access. Promoted to HOT on read.
COLD  →  Archive      Cheap storage. Rarely accessed. Promoted on demand.
```

The decay worker runs in the background. You never manage tiers manually.

---

## Memory types

```python
from langmemory import LangMemory, MemoryType

memory = LangMemory()

# Episodic — specific events (decays at normal rate)
memory.insert("User asked about deploying to Lambda", memory_type=MemoryType.EPISODIC)

# Semantic — general knowledge (decays slowly)
memory.insert("User prefers minimal dependencies", memory_type=MemoryType.SEMANTIC)

# Procedural — how-to knowledge
memory.insert("To deploy: run make deploy then push to main", memory_type=MemoryType.PROCEDURAL)

# Working — transient context (decays fast)
memory.insert("User is currently looking at the auth module", memory_type=MemoryType.WORKING)
```

---

## Integrity verification

Every memory is Merkle-chained. Any modification to any node breaks its hash.

```python
report = memory.verify_integrity()

print(report.is_valid)          # True
print(report.total_nodes)       # 42
print(report.corrupted_nodes)   # []
print(report.root_hash.hex())   # deterministic fingerprint of entire memory state
```

Store the root hash externally to detect tampering across sessions. This is the property no other LLM memory system provides.

---

## Concept-scoped retrieval

```python
# Index memories under concept paths
memory.insert(
    "Use async generators for streaming responses",
    concepts=["Python/async", "API/streaming"]
)

# Retrieve everything under a concept subtree
results = memory.retrieve("streaming", concept_filter="Python/async")
```

---

## Embedder backends

The default embedder is a deterministic stub — no API key, no network, full functionality.

Switch backends without changing anything else:

```python
from langmemory import LangMemory, ChainConfig

# OpenAI (pip install openai)
memory = LangMemory(ChainConfig(embedder_backend="openai"))

# Local / offline (pip install sentence-transformers)
memory = LangMemory(ChainConfig(embedder_backend="sentence-transformers"))
```

---

## LangChain integration

```python
from langmemory import LangMemory
from langmemory.adapters.langchain import LangChainMemory
from langchain.chains import ConversationChain

memory = LangChainMemory(chain=LangMemory())
chain = ConversationChain(llm=llm, memory=memory)
```

---

## Configuration

```python
from langmemory import LangMemory, ChainConfig

memory = LangMemory(ChainConfig(
    hot_tier_limit=500,          # nodes kept in RAM
    semantic_weight=0.6,         # retrieval score weights
    importance_weight=0.3,
    recency_weight=0.1,
    default_decay_factor=0.01,   # importance decay per idle day
    bloom_capacity=50_000,
))
```

---

## Stats

```python
memory.stats()
# {
#   "hot_count": 38,
#   "warm_count": 0,
#   "cold_count": 0,
#   "total_indexed": 38,
#   "bloom_fp_rate": 0.0002,
#   "root_hash": "a3f9...",
#   "skip_list_size": 38
# }
```

---

## Why not RAG?

RAG is retrieval over documents. LangMemory is a memory runtime for agents.

| | RAG | LangMemory |
|---|---|---|
| Structure | Flat vector index | Tiered graph with integrity chain |
| Deduplication | None | Bloom filter at insert time |
| Temporal queries | Not supported | Skip list range queries |
| Importance decay | Not supported | Background decay worker |
| Integrity | None | Merkle chain, verifiable root hash |
| Dependencies | Heavy (vector DB, embedder) | Zero required |
| Complexity guarantees | Undocumented | Every operation documented |

---

## License

MIT — Puneet Singh
