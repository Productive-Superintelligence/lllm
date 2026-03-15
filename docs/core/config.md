# Configuration & Discovery

LLLM favors composition over hard-coded imports. Auto-discovery plus lightweight YAML allows an entire system to be wired together without editing Python entry points.

## Configuration Files

1. **`config/<system>/default.yaml`** – runtime settings consumed by agents and systems.
   - `name`, `log_type`, `log_dir`, `ckpt_dir`, randomness controls.
   - `agent_configs` describing each model (name, prompt path, temperature, `max_completion_tokens`, optional `api_type`).
   - Execution guards such as `max_exception_retry`, `max_interrupt_times`, `max_llm_recall`.
   - Proxy activation and deploy toggles.
2. **`lllm.toml`** – discovery manifest read by `lllm.config.find_config_file` and `lllm.discovery.auto_discover`.

Example (`template/lllm.toml`):

```toml
[prompts]
folders = ["system/agent/prompts"]

[proxies]
folders = ["system/proxy/modules"]
```

Place this file at the project root or point to it via `$LLLM_CONFIG`. The loader searches parents recursively, so you can run tools from subdirectories without losing context.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `LLLM_CONFIG` | Absolute path to a config file or folder; overrides auto-detection. |
| `LLLM_AUTO_DISCOVER` | Set to `0`, `false`, or `no` to skip auto-discovery (manual registration only). |
| `TMP_DIR` | Overrides the default temp/cache directory used by utils and error logging. |

## Auto-Discovery Workflow

`lllm.auto_discover()` runs when the package is imported:

1. Resolve the config path (`LLLM_CONFIG`, explicit argument, or nearest `lllm.toml`).
2. Load TOML, collect prompt/proxy folders relative to the file.
3. Import every `.py` file under those folders (excluding `__init__` and private files).
4. Register each `Prompt` (keyed by `namespace/module.prompt.path`).
5. Register each `BaseProxy` subclass (keyed by `_proxy_path` or `<namespace>/<class>`).

Because registration happens at import time, simply adding a new prompt module to the folder makes it available across the repo without touching central registries.

## YAML Tips

- Keep secrets (API keys) out of the YAML and load them via environment variables inside your system/agent code.
- Use multiple YAML files (e.g., `config/prod.yaml`) and load the desired profile before building a system.
- Version-control template configs but store user-specific overrides elsewhere; the CLI scaffold already sets up a namespaced config folder.

## Disabling Discovery

When packaging LLLM as a reusable library, you may want to opt out of auto-imports. Set `LLLM_AUTO_DISCOVER=0`, then register prompts and proxies manually:

```python
from lllm import register_prompt, register_proxy
register_prompt(my_prompt)
register_proxy("custom/my_proxy", MyProxy)
```

This pattern is useful for unit tests or environments where dynamic imports are restricted.
