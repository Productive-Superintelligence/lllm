# Project Reference

Quick reference for naming conventions, folder layout, and configuration tips. For a step-by-step walkthrough of building a package, see [Tutorial: Build a Full Package](building-agents.md).

---

## Naming Conventions

| Resource type | Location | Naming |
|---|---|---|
| System prompts | `prompts/<agent>_system.md` | `<agent>_system` |
| User prompt templates | `prompts/<agent>_user.md` | `<agent>_user` |
| Agent configs | `configs/<tactic_name>.yaml` | `<tactic_name>` |
| Tactic classes | `tactics/<tactic_name>.py` | class `<TacticName>` |
| Proxy classes | `proxies/<service>_proxy.py` | class `<Service>Proxy` |

---

## Recommended Folder Layout

```
my_project/
├── lllm.toml
├── prompts/
├── configs/
│   ├── default.yaml            # main config
│   ├── experiments/            # variant configs (base: default)
│   └── vendor/                 # pinned dependency configs
├── tactics/
├── proxies/
├── runs/                       # session checkpoints (auto-created)
└── main.py
```

For multi-package projects:

```
workspace/
├── shared_lib/
│   ├── lllm.toml
│   └── prompts/
└── my_project/
    ├── lllm.toml               # [dependencies] packages = ["../shared_lib"]
    └── ...
```

---

## Session & Log Output

```
runs/
├── sessions/           ← checkpoint files per tactic run
│   └── 20250316_142301_a3f7b2c1.json
└── logs/               ← if using local_store()
```

The `ckpt_dir` argument to `build_tactic()` controls where checkpoints land. Pass `ckpt_dir=None` to disable checkpointing.

---

## Configuration Tips

**Environment variables override TOML:**

```bash
export LLLM_CONFIG=/path/to/custom/lllm.toml
```

**Multiple environments:**

```bash
LLLM_CONFIG=lllm.prod.toml python main.py
```

**Validate your config before running:**

```python
from lllm import resolve_config
config = resolve_config("my_tactic")
print(config)  # inspect the merged config dict
```
