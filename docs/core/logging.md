# Logging & Printing

LLLM provides two orthogonal systems:

- **Session logging** — persist every `TacticCallSession` to a backend store; query and retrieve them later by tactic identity, tags, or time range.
- **Terminal logging** — route internal diagnostic messages through Python's standard `logging` module with ANSI colours.

---

## Session Logging

### Core idea

Tactics are the top-level abstraction in LLLM, so the unit of logging is the `TacticCallSession` — one complete invocation of a tactic. Every call is automatically saved if you hand a `LogStore` to the tactic.

### Tactic identity: the stable `pkg::name` key

Every session carries a **tactic identity** that answers "which tactic produced this session?" unambiguously. LLLM uses the format:

```
{package_name}::{tactic_name}
```

For example: `my_pkg::researcher`.

#### Why not use the full runtime path?

The runtime path (`my_pkg.tactics:folder/researcher`) encodes where the tactic lives inside the package — its directory, any `under` prefix from lllm.toml, etc. That's useful for *accessing* a tactic at runtime but is a poor physical storage key because:

| Structural change | Effect on runtime path | Effect on `pkg::name` |
|---|---|---|
| Move tactic file to subfolder | Changes | **Stable** |
| Add `under v2` in lllm.toml | Changes | **Stable** |
| Add/change/remove an alias | Changes aliases | **Stable** |
| Rename `[package] name` | Changes | Changes |
| Rename `Tactic.name` | Changes | Changes |

The last two rows are *semantic* changes — renaming the package or the tactic — and are expected to be deliberate breaking changes. File layout and `under` prefixes are structural, and restructuring should never corrupt historical logs.

**The solution:** Store sessions under `pkg::name` (stable). Resolve any runtime path or alias to this stable key at query time using the attached runtime. The runtime itself is ephemeral; the DB is permanent.

```
Physical DB key:    my_pkg::researcher             ← stored once, never changes
Runtime path:       my_pkg.tactics:v2/researcher   ← changes freely with toml
Alias path:         alias.tactics:v2/researcher    ← resolves → stable key at query time
```

#### What if I rename the package?

Old sessions remain under `old_pkg::researcher`. New sessions go under `new_pkg::researcher`. If you need continuity, you have two options:

1. **Migration script**: iterate over old sessions, re-save them under the new key, delete the old.
2. **Keep the package name stable**: treat `[package] name` as a permanent identity, like a Python package name. This is the recommended approach.

#### Tactics used without the package system

When a tactic is constructed directly (not via `build_tactic` or package discovery), there is no namespace to extract a package name from. In that case `tactic_path` falls back to the bare `Tactic.name` (e.g., `"researcher"`). Sessions are still saved and queryable by that bare name.

---

### Creating a LogStore

`LogStore` wraps a `LogBackend` and provides the full query API. Three convenience factories cover the common cases:

```python
from lllm.logging import local_store, sqlite_store, noop_store
from lllm import get_default_runtime

rt = get_default_runtime()

# --- local files (one file per session, human-readable JSON tree) ---
store = local_store("~/.lllm/logs", runtime=rt)

# --- SQLite (single portable file, atomic writes) ---
store = sqlite_store("~/.lllm/logs.db", runtime=rt)

# --- dry-run / tests (silently discards everything) ---
store = noop_store()
```

The `runtime` argument enables alias resolution when querying by tactic path (see [Querying by tactic path](#querying-by-tactic-path)).

You can also construct them explicitly:

```python
from lllm.logging import LogStore, LocalFileBackend, SQLiteBackend

# Multiple partitions in one directory — each project stays isolated
store_a = LogStore(LocalFileBackend("~/.lllm/logs"), partition="project-a", runtime=rt)
store_b = LogStore(LocalFileBackend("~/.lllm/logs"), partition="project-b", runtime=rt)

# Multiple partitions in one SQLite file
store = LogStore(SQLiteBackend("~/.lllm/all.db"), partition="prod", runtime=rt)
```

#### Backend comparison

| Backend | Best for | Storage |
|---|---|---|
| `LocalFileBackend` | Development, human inspection of raw JSON | One file per session under a directory tree |
| `SQLiteBackend` | Production, atomic writes, concurrent access | Single `.db` file |
| `NoOpBackend` | Tests, CI, dry-run workflows | Nothing written |

### Attaching a store to a tactic

Pass `log_store=` at construction time. All subsequent calls are saved automatically.

```python
from lllm import get_default_runtime
from lllm.logging import sqlite_store
from lllm.core.tactic import build_tactic
from lllm.core.config import resolve_config

rt = get_default_runtime()
store = sqlite_store("~/.lllm/logs.db", partition="my-project", runtime=rt)

# build_tactic resolves pkg::name from the runtime and embeds it in every session
config = resolve_config("default")
tactic = build_tactic(config, ckpt_dir="./runs", log_store=store)
```

Or when subclassing `Tactic` directly:

```python
from lllm.core.tactic import Tactic

class MyTactic(Tactic):
    name = "my_tactic"
    agent_group = ["analyst"]

    def call(self, task, **kwargs):
        ...

tactic = MyTactic(config, ckpt_dir="./ckpt", log_store=store)
```

If no store is configured the tactic emits a `UserWarning` once per instance and continues normally.

### Saving sessions with tags

```python
# Basic call — session auto-saved under "my_pkg::my_tactic"
result = tactic(task)

# Call with tags for filtering later
result = tactic(
    task,
    tags={
        "experiment": "exp-001",
        "dataset":    "arxiv-2024",
        "split":      "val",
    },
    metadata={"user": "alice", "git_sha": "abc1234"},
)
```

`tags` are indexed for filtering. `metadata` is arbitrary extra data stored alongside the session but not used for filtering.

---

## Querying by tactic path

`list_sessions(tactic_path=...)` accepts **any** path form the runtime understands and resolves them all to the same stable key:

```python
# All of the following are equivalent when the runtime is attached:
store.list_sessions(tactic_path="my_pkg::researcher")              # stable key (direct)
store.list_sessions(tactic_path="my_pkg.tactics:folder/researcher") # full runtime path
store.list_sessions(tactic_path="my_pkg:folder/researcher")         # package-qualified
store.list_sessions(tactic_path="alias_pkg:folder/researcher")      # any alias
store.list_sessions(tactic_path="researcher")                       # bare name (default ns)
```

The resolution chain:
1. `store._resolve_tactic_path(path)` → calls `runtime.get_node(path, resource_type="tactic")`
2. Extracts `package_name` from `node.namespace` (e.g. `"my_pkg.tactics"` → `"my_pkg"`)
3. Extracts `tactic_name` from `node.value.name` (the class attribute)
4. Returns `"my_pkg::researcher"`

If no runtime is attached, or the path is not in the registry, the path is compared literally.

### Listing and filtering sessions

```python
import datetime as dt

# All sessions (newest first, up to 100)
summaries = store.list_sessions()

# Filter by tactic (any path form)
summaries = store.list_sessions(tactic_path="my_pkg:folder/researcher")

# Filter by tags — all specified tags must match exactly
summaries = store.list_sessions(tags={"experiment": "exp-001"})

# Filter by state: "success", "failure", "running"
failures = store.list_sessions(state="failure")

# Filter by time range
summaries = store.list_sessions(
    after=dt.datetime(2025, 1, 1),
    before=dt.datetime(2025, 6, 1),
)

# Combine all filters
summaries = store.list_sessions(
    tactic_path="my_pkg:folder/researcher",
    tags={"experiment": "exp-001"},
    state="success",
    limit=50,
)

# Inspect a summary
for s in summaries:
    print(s.session_id)        # "20250316_142301_a3f7c2b1"
    print(s.tactic_path)       # "my_pkg::researcher"   ← stable key
    print(s.tactic_name)       # "researcher"           ← simple name
    print(s.state)             # "success"
    print(s.total_cost)        # 0.0023 (USD)
    print(s.agent_call_count)  # 4
    print(s.timestamp)         # ISO-8601
    print(s.tags)
```

`SessionSummary` fields:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Unique ID, format `YYYYMMDD_HHMMSS_<hex8>` |
| `tactic_name` | `str` | Simple `Tactic.name` attribute |
| `tactic_path` | `str` | Stable key `pkg::name`, e.g. `"my_pkg::researcher"` |
| `state` | `str` | `"success"`, `"failure"`, or `"running"` |
| `total_cost` | `float` | Total USD cost (sum of all agent calls) |
| `agent_call_count` | `int` | Number of agent LLM calls |
| `timestamp` | `str` | ISO-8601 |
| `tags` | `Dict[str, str]` | Tags attached at call time |

### Loading a full session

```python
# Load the TacticCallSession object
session = store.load_session(session_id)

print(session.tactic_name)          # "researcher"
print(session.tactic_path)          # "my_pkg::researcher"
print(session.state)                # "success" | "failure"
print(session.total_cost)           # InvokeCost
print(session.agent_call_count)
print(session.delivery)             # return value of tactic.call()
print(session.error)                # error string if failed
print(session.error_traceback)      # full traceback string if failed

# Browse all agent call sessions
for agent_name, agent_sessions in session.agent_sessions.items():
    for agent_sess in agent_sessions:
        print(agent_name, agent_sess.state, agent_sess.cost)
        # agent_sess.delivery → the Message returned
        # agent_sess.interrupts → tool calls at each interrupt step
        # agent_sess.invoke_traces → raw InvokeResult per step
```

### Loading a session with metadata

```python
record = store.load_session_record(session_id)

print(record.session_id)
print(record.timestamp)
print(record.tags)
print(record.metadata)
print(record.traceback)   # full traceback string, or None if succeeded

session = record.session  # full TacticCallSession
```

### Inspecting failures

```python
failures = store.list_sessions(
    tactic_path="my_pkg:folder/researcher",
    state="failure",
)
for s in failures:
    record = store.load_session_record(s.session_id)
    print(f"=== {s.tactic_path} @ {s.timestamp} ===")
    print(record.traceback)
```

Save standalone errors (outside any tactic):

```python
try:
    ...
except Exception as e:
    error_id = store.save_error(e, context={"step": "preprocessing", "file": path})
```

### Cost summary

```python
summary = store.export_cost_summary()
print(summary["total_cost_usd"])
print(summary["session_count"])

summary = store.export_cost_summary(
    tactic_path="my_pkg:folder/researcher",
    tags={"experiment": "exp-001"},
    after=dt.datetime(2025, 3, 1),
)
for row in summary["sessions"]:
    print(row["session_id"], row["cost_usd"])
```

### Exporting and deleting

```python
json_str = store.export_session(session_id, format="json")
store.delete_session(session_id)
```

---

## Tag System — Best Practices

Tags are `str → str` dictionaries indexed for exact-match filtering.

### Recommended vocabulary

```python
# Experiment / run tracking
tags = {"experiment": "ablation-001", "run": "run-003", "variant": "no-retrieval"}

# Dataset splits
tags = {"dataset": "pubmed-2024", "split": "test", "subset": "cardiology"}

# Multi-stage pipelines
tags = {"pipeline": "document-qa", "stage": "extraction", "doc_id": "doc-00127"}

# Production / CI
tags = {"env": "prod", "version": "v2.3.1", "requester": "user-alice"}
```

### Tag composition pattern

Build tags incrementally so every level is independently filterable:

```python
BASE = {"project": "doc-qa", "model": "gpt-4o"}

for exp in ["baseline", "rag-v1", "rag-v2"]:
    for doc_id in document_ids:
        tactic(make_task(doc_id), tags={**BASE, "experiment": exp, "doc_id": doc_id})

# Query at any granularity:
store.list_sessions(tags={"project": "doc-qa"})
store.list_sessions(tags={"project": "doc-qa", "experiment": "baseline"})
store.list_sessions(tags={"project": "doc-qa", "doc_id": "doc-00127"})
```

### Partitions for project isolation

```python
store_a = local_store("~/.lllm/logs", partition="project-a", runtime=rt)
store_b = local_store("~/.lllm/logs", partition="project-b", runtime=rt)
# Sessions in each partition are invisible to the other
```

### Tags vs metadata

- **tags** — short, machine-readable values for filtering.
- **metadata** — rich context you want to retrieve but not filter on.

```python
tactic(
    task,
    tags={"experiment": "exp-001", "split": "val"},
    metadata={"git_sha": "abc1234", "input_file": "/data/val.jsonl", "config": config},
)
```

---

## Terminal Logging

LLLM uses Python's standard `logging` module internally. All loggers live under the `lllm` hierarchy:

| Logger | Covers |
|---|---|
| `lllm` | Root — captures everything |
| `lllm.tactic` | Tactic start/completion/failure, session persistence |
| `lllm.agent` | Agent open/respond lifecycle |
| `lllm.invoker` | LLM API call retries, parse errors |
| `lllm.config` | Package discovery, missing config warnings |
| `lllm.proxy` | Proxy manager registration |
| `lllm.sandbox` | Jupyter/sandbox lifecycle |
| `lllm.cua` | Computer-use agent events |

### Quick setup

```python
from lllm.logging import setup_logging

setup_logging()                                   # INFO, colours on
setup_logging("DEBUG")                            # show tool traces, etc.
setup_logging("WARNING", color=False)             # plain output for CI
setup_logging("INFO", fmt="%(asctime)s  %(levelname)s  %(name)s — %(message)s")
```

Colour scheme:

| Level | Colour |
|---|---|
| DEBUG | Dark gray |
| INFO | Green |
| WARNING | Yellow |
| ERROR | Red |
| CRITICAL | Bold red |

### Using stdlib logging directly

```python
import logging

# Fine-grained control
logging.getLogger("lllm").setLevel(logging.WARNING)
logging.getLogger("lllm.tactic").setLevel(logging.INFO)

# File handler for production
handler = logging.FileHandler("lllm.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
logging.getLogger("lllm").addHandler(handler)
```

### Level guide

| Level | What you see |
|---|---|
| `WARNING` (default) | Rate limit warnings, missing config, recoverable errors |
| `INFO` | Tactic start/completion, cost per call |
| `DEBUG` | Every agent respond, tool call traces, prompt rendering |

### Silencing specific subsystems

```python
logging.getLogger("lllm").setLevel(logging.ERROR)         # mostly silent
logging.getLogger("lllm.invoker").setLevel(logging.ERROR) # mute retry noise
logging.getLogger("lllm").setLevel(logging.CRITICAL + 1)  # fully silent
```

### Custom ColoredFormatter

```python
import logging, sys
from lllm.logging import ColoredFormatter

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter("%(levelname)-8s %(name)s — %(message)s"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)
```

---

## Custom Backends

Implement `LogBackend` to connect any KV store. The interface has four methods operating on `str` keys and `bytes` values:

```python
from lllm.logging import LogBackend, LogStore
from typing import List, Optional


class MyBackend(LogBackend):

    def put(self, key: str, data: bytes) -> None:
        """Write *data* under *key*, overwriting any existing entry."""
        ...

    def get(self, key: str) -> Optional[bytes]:
        """Return the bytes stored under *key*, or ``None`` if missing."""
        ...

    def list_keys(self, prefix: str = "") -> List[str]:
        """Return all keys that start with *prefix* (empty = all keys)."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key* (silently succeed if it does not exist)."""
        ...
```

Keys are forward-slash-separated strings, e.g. `"default/sessions/20250316_a3f7c2b1"`. You do not need to interpret their structure — `LogStore` owns the layout.

### Redis example

```python
import redis
from lllm.logging import LogBackend, LogStore
from typing import List, Optional


class RedisBackend(LogBackend):
    def __init__(self, client: redis.Redis, key_prefix: str = "lllm"):
        self._r = client
        self._prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def put(self, key: str, data: bytes) -> None:
        self._r.set(self._k(key), data)

    def get(self, key: str) -> Optional[bytes]:
        return self._r.get(self._k(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        pattern = self._k(prefix) + "*"
        strip = len(self._prefix) + 1
        return [k.decode()[strip:] for k in self._r.keys(pattern)]

    def delete(self, key: str) -> None:
        self._r.delete(self._k(key))


store = LogStore(RedisBackend(redis.Redis()), partition="prod", runtime=rt)
```

### Firebase / Firestore example

```python
from google.cloud import firestore
from lllm.logging import LogBackend, LogStore
from typing import List, Optional


class FirestoreBackend(LogBackend):
    def __init__(self, collection: str = "lllm_logs"):
        self._db = firestore.Client()
        self._col = self._db.collection(collection)

    def _doc_id(self, key: str) -> str:
        return key.replace("/", "__")  # Firestore IDs can't contain "/"

    def put(self, key: str, data: bytes) -> None:
        self._col.document(self._doc_id(key)).set({"data": data, "key": key})

    def get(self, key: str) -> Optional[bytes]:
        snap = self._col.document(self._doc_id(key)).get()
        return snap.get("data") if snap.exists else None

    def list_keys(self, prefix: str = "") -> List[str]:
        return [d.get("key") for d in self._col.stream()
                if d.get("key", "").startswith(prefix)]

    def delete(self, key: str) -> None:
        self._col.document(self._doc_id(key)).delete()


store = LogStore(FirestoreBackend("my_project_logs"), partition="prod", runtime=rt)
```

### Design notes for custom backends

- **Atomicity**: `put` should be atomic or use WAL/transactions where available. Non-atomic writes can corrupt index entries.
- **`list_keys` efficiency**: `list_sessions` calls `list_keys(prefix)` then `get` for every returned key. For large stores, implement prefix indexing or a secondary index to avoid a full scan.
- **Concurrency**: If multiple processes write to the same backend, ensure `put` is safe under concurrent access. `SQLiteBackend` uses WAL mode; `LocalFileBackend` relies on OS filesystem atomicity.
- **Serialisation**: All data passed to `put` is already JSON-encoded `bytes`. Store it as a blob, base64, or JSON document — as long as `get` returns the same bytes.

---

## Complete Example

```python
import datetime as dt
from lllm import load_runtime
from lllm.logging import setup_logging, sqlite_store
from lllm.core.tactic import build_tactic
from lllm.core.config import resolve_config

# 1. Load runtime from lllm.toml
rt = load_runtime()

# 2. Configure terminal output
setup_logging("INFO")

# 3. Create a persistent store bound to the runtime.
#    Sessions are stored under the stable key "my_pkg::researcher",
#    regardless of file layout or aliases in lllm.toml.
store = sqlite_store("~/.lllm/my-project.db", partition="experiment-001", runtime=rt)

# 4. Build tactic — pkg::name is resolved and embedded in every session
config = resolve_config("default")
tactic = build_tactic(config, ckpt_dir="./runs", log_store=store)

# 5. Run with tags
for i, doc in enumerate(documents):
    tactic(
        doc,
        tags={"experiment": "experiment-001", "doc_id": f"doc-{i:04d}", "split": "test"},
        metadata={"source_path": doc.path},
    )

# 6. All of the following query the same sessions (resolved to "my_pkg::researcher"):
store.list_sessions(tactic_path="my_pkg::researcher")              # direct stable key
store.list_sessions(tactic_path="my_pkg.tactics:research/researcher")  # runtime path
store.list_sessions(tactic_path="my_pkg:research/researcher")      # short form
store.list_sessions(tactic_path="researcher")                       # bare name

# If you later add "under v2" in lllm.toml, the tactic moves to
# my_pkg.tactics:v2/researcher in the runtime, but all of the above
# still query the same stored sessions because they resolve to the same
# stable key "my_pkg::researcher".

# 7. Inspect failures
failures = store.list_sessions(
    tactic_path="my_pkg:research/researcher",
    state="failure",
)
for s in failures:
    record = store.load_session_record(s.session_id)
    print(f"[{s.tactic_path}] failed @ {s.timestamp}")
    print(record.traceback)

# 8. Cost report
summary = store.export_cost_summary(
    tactic_path="my_pkg:research/researcher",
    tags={"experiment": "experiment-001"},
)
print(f"{summary['session_count']} sessions  ${summary['total_cost_usd']:.4f}")

# 9. Drill into one session
session = store.load_session(failures[0].session_id)
for agent_name, calls in session.agent_sessions.items():
    for call in calls:
        print(f"  {agent_name}: {call.state}  cost={call.cost}")
```
