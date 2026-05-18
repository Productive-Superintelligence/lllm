# Unit Tests

**669 tests** across 13 test files. They do not require API keys; context-window tests require `litellm`, and cache tests need a writable LLLM cache directory.

Run the full suite:

```bash
pytest tests/units/ -q
```

Run a single file:

```bash
pytest tests/units/test_prompt.py -v
```

---

## Test files

### `test_const.py` — 41 tests
Core enumerations and data structures (`lllm/core/const.py`).

| Class | What it tests |
|---|---|
| `TestRolesEnum` | `Roles` str-enum values; `system` → `developer` mapping for O-series models |
| `TestInvokersEnum` | `Invokers` enum membership |
| `TestModalitiesEnum` | `Modalities` all values and count |
| `TestAPITypesEnum` | `APITypes` values |
| `TestLLMSideRoles` | `LLM_SIDE_ROLES` set contents |
| `TestParseError` | `ParseError` construction, inheritance, string representation |
| `TestFunctionCall` | `FunctionCall` construction, equality, `is_repeated()`, `success` property, `__str__` |
| `TestInvokeCost` | `InvokeCost` defaults, `__add__`, all token fields, rate display in `__str__` |
| `TestInvokeResult` | `InvokeResult` construction, cost aggregation, error list |

---

### `test_dialog.py` — 61 tests
Dialog, message, and conversation tree structures (`lllm/core/dialog.py`).

| Class | What it tests |
|---|---|
| `TestTokenLogprob` | Pydantic model construction, defaults, nested `top_logprobs`, extra fields |
| `TestMessage` | Content types (str/list), role handling, `is_function_call`, `sanitized_name`, logprobs coercion, `cost` property with usage details (including `None`-safe arithmetic), dict roundtrip |
| `TestDialogTreeNode` | `add_child`, parent/child refs, `depth`, `subtree_ids`, dict roundtrip |
| `TestDialog` | Construction defaults, `put_text`, `put_prompt`, `put_image` (PIL, base64, invalid), `fork` with `last_n`, nested fork depth, tree linking, cost aggregation, `overview` truncation |

---

### `test_prompt.py` — 91 tests
Prompt rendering, extension, parsing, and the `@tool` decorator (`lllm/core/prompt.py`).

| Class | What it tests |
|---|---|
| `TestPromptBasic` | Static prompts, template variable detection, rendering, missing-var errors, literal double-braces |
| `TestPromptExtend` | `extend()` inherits metadata, requires new path, overrides fields |
| `TestDefaultTagParser` | XML tag extraction, markdown block extraction, signal tags, multi-match, missing-required-tag error (`ParseError`) |
| `TestPromptWithParser` | `parse()` delegates to parser, `__call__` renders and parse pipeline |
| `TestToolDecorator` | `@tool()` creates `Function` objects, JSON schema generation, execution, docstring extraction |

---

### `test_utils.py` — 50 tests
Utility functions (`lllm/utils/`).

| Class | What it tests |
|---|---|
| `TestLoadJson` / `TestSaveJson` | JSON load/save, missing-file defaults, UTF-8, overwrite |
| `TestFindXmlBlocks` | Single/multiple/nested XML block extraction, empty tags, no-match |
| `TestFindMdBlocks` | Markdown fenced block extraction (` ```python `, generic) |
| `TestFindAllXmlTagsSorted` | All-tag scan returning sorted dict |
| `TestHtmlCollapse` | `html_collapse()` wraps content in `<details>` |
| `TestDirectoryTree` | `directory_tree()` formatting, depth limit, files vs dirs-only mode |
| `TestCheckItem` | Required-key validation, type checking, `ParseError` on mismatch |
| `TestIsOpenAIRateLimitError` | Rate-limit error string detection |
| `TestCacheUtilities` | Cache key determinism, save/load, non-existent returns `None` |

---

### `test_proxy.py` — 51 tests
Proxy base class and endpoint decorator (`lllm/proxies/base.py`).

| Class | What it tests |
|---|---|
| `TestBaseProxyConstruction` | Instantiation, `_proxy_path`, `_proxy_name`, `_proxy_description` defaults |
| `TestEndpointDecorator` | `@BaseProxy.endpoint(...)` registers metadata on method, params/response schema |
| `TestProxyExecution` | Calling endpoint methods, return values, error handling |
| `TestProxyDiscovery` | Multiple endpoints on one proxy, endpoint iteration |

---

### `test_tactic_session.py` — 35 tests
Tactic session tracking and agent wrapping (`lllm/core/tactic.py`).

| Class | What it tests |
|---|---|
| `TestAgentCallSession` | Session construction, `record_agent_call`, `record_sub_tactic_call`, cost aggregation, `success()`, `summary()`, Pydantic serialization |
| `TestTacticCallSession` | Multi-agent recording, sub-tactic costs, state transitions, summary structure |
| `TestTrackedAgent` | `__getattr__`/`__setattr__` delegation, `respond()` recording, `return_session=True` |
| `TestTacticRegistration` | `@Tactic` subclass auto-registers by `name`, `get_nonexistent` raises |
| `TestTacticAutoRegistration` | Subclass with `name` attribute is auto-registered on class definition |

---

### `test_tactic_tool.py` — 25 tests
Tactic tools, package-discovered regular tools, proxy refs in agent config, and shared tool URL resolution (`lllm/core/tactic_tool.py`, `lllm/core/config.py`, `lllm/proxies/base.py`).

| Class | What it tests |
|---|---|
| `TestTacticPromptToolRefs` | `@tactictool`, full/bare/shorthand tactic URLs, URL fragments, fallback warnings, missing config errors |
| `TestRegisteredFunctionToolRefs` | Regular `Function` tool URLs, mixed direct/URL/declaration prompt styles, prompt declaration auto-binding, `[tools]` section discovery, package-local bare refs, ambiguity between tool and tactic shorthand |
| `TestTacticProxyEndpoints` | Tactic-backed proxy endpoints, API directory exposure, `ProxyManager` dispatch, imperative registration |
| `TestAgentConfigTools` | Global/per-agent `tools`, mixed regular/tactic/proxy refs, proxy refs enabling `query_api_doc` / `run_python` |

---

### `test_package_system.py` — 64 tests
Low-level package system primitives using temp directories (`lllm/core/resource.py`, `lllm/core/runtime.py`, `lllm/core/config.py`).

| Class | What it tests |
|---|---|
| `TestResourceNode` | Eager vs lazy nodes, qualified key format, namespace-less nodes |
| `TestPackageInfo` | `effective_name` with/without alias |
| `TestRuntime` | Default namespace resolution, `has()`, `keys()` filtering, `not found` error, `pkg:shorthand`, register/get, `reset()`, type mismatch error |
| `TestParseDependencyEntry` | `dep plain`, `dep "x as y"`, `{path, alias}`, `{path, as}` table forms |
| `TestParsePathEntry` | `path plain`, `"path under prefix"`, `{path, prefix}`, `{path, under}` table forms |
| `TestLoadToml` | Valid TOML load, missing file returns `None`, `_config_path` injected |
| `TestLoadPackageTempDir` | Full package load from temp dir: prompts registered, configs lazy, dependencies, custom sections, cycle detection, `load_cwd_fallback` |
| `TestResolveConfig` | Base inheritance, deep-merge semantics, circular detection |
| `TestVendorConfig` | No-override passthrough, override merging |
| `TestAgentSpec` | `from_config()` parsing, inline vs path system prompts, unknown keys → model_args, missing required fields raise |
| `TestParseAgentConfigs` | global merge, multi-agent, missing agent raises |

---

### `test_package_integration.py` — 107 tests
End-to-end package loading using **real package fixtures** on disk (`tests/test_cases/packages/`).

| Class | Package(s) | What it tests |
|---|---|---|
| `TestPkgAlpha` | `pkg_alpha` | Prompt registration (flat + nested subfolders), template vars, rendering, missing-var errors, literal braces, parser (XML tags, signal tags), config lazy loading, 2-level config inheritance (`default` → `production`, `development`), `parse_agent_configs`, prompt metadata |
| `TestPkgBeta` | `pkg_beta` → `pkg_alpha` | 2-level dependency, default namespace, cross-package prompt access, config content, agent spec parsing, prompt rendering, debate parser |
| `TestPkgGamma` | `pkg_gamma` | Custom sections (`assets`, `data`), JSON/YAML structured loaders, PNG binary bytes, nested assets, hidden-file exclusion, `__pycache__` exclusion, `file_path` metadata |
| `TestPkgEmpty` | `pkg_empty` | Empty package: no resources, no prompts, no configs |
| `TestPkgPrefixed` | `pkg_prefixed` | Under-prefix path (`vendor_prompts/ under tools`), correct key format, rendering, no-access without prefix |
| `TestCycleDetection` | `pkg_cycle_a` ↔ `pkg_cycle_b` | Cycle terminates without hang, both packages registered exactly once, load from either side |
| `TestPkgDelta` | `pkg_delta` | Proxies and prompts together |
| `TestLoadResourceWithRealPackages` | `pkg_gamma` | `load_resource()` full URL, section-only, missing colon raises, nonexistent raises |
| `TestLoadRuntimeWithRealPackage` | `pkg_alpha` | `load_runtime()` from TOML, named runtime, `get_runtime()` nonexistent raises |
| `TestVendorConfigWithRealPackages` | `pkg_alpha` | `vendor_config()` no overrides, with overrides (deep-merge), base chain resolved first |
| `TestMultiplePackagesInOneRuntime` | `pkg_alpha` + `pkg_gamma` | Coexistence, first-loaded namespace wins, no resource collision |
| `TestPackageInfoFromRealPackages` | `pkg_alpha`, `pkg_gamma` | `PackageInfo` fields, empty version, `effective_name` |
| `TestRuntimeResetWithPackages` | `pkg_alpha` | `reset()` clears all, reload after reset works |
| `TestKeysFiltering` | `pkg_alpha` | `keys("prompt")` and `keys("config")` are disjoint, all keys superset |

---

### `test_advanced_package.py` — 54 tests
Complex package management scenarios using real fixtures and temp packages.

| Class | Scenario |
|---|---|
| `TestFourLevelDependencyChain` | `pkg_chain_a → b → c → d`: all 4 packages registered, prompts/configs accessible at every level, default namespace is root |
| `TestDiamondDependency` | `root → left + right → shared`: shared package loaded exactly once via `_loaded_package_paths` deduplication, no duplicate keys |
| `TestPackageLoadIdempotency` | Calling `load_package` twice for the same path is a no-op (resource count unchanged, `_loaded_package_paths` not grown) |
| `TestMissingDependencyWarning` | Broken dependency path emits `RuntimeWarning` and does not raise; package itself still registers |
| `TestInvalidToml` | Malformed TOML raises an exception on load |
| `TestRegisterOverwrite` | `overwrite=False` raises `ValueError` on collision; `overwrite=True` replaces silently |
| `TestRegisterConfigLazyLoader` | `register_config(loader=fn)` defers call until first access, then caches (loader called exactly once) |
| `TestFindConfigFileEnvVar` | `LLLM_CONFIG` env var pointing at a file, a directory, or a nonexistent path |
| `TestGetProxyAndTacticQualifiedKey` | `get_proxy()` with full qualified key, bare key via default namespace, multiple proxies in one file |
| `TestVendorConfigMultiLevel` | `vendor_config` resolves full inheritance chain then applies overrides; result can be re-registered; circular `base` chain raises `ValueError` |
| `TestPackageAliasing` | Dependency declared as `{path, alias}` registers resources under the alias namespace |
| `TestSameSectionNameNoCollision` | Two packages with the same section name (`prompts`, `configs`) coexist without overwriting each other |
| `TestRuntimeKeysSortedFiltered` | `keys()` returns sorted list; typed filters are disjoint; unknown type returns `[]` |
| `TestThreeLevelConfigInheritance` | 3-level `base` chain with deep-merge: each level's scalar overrides win, dict keys merge, non-overridden keys propagate from base |

---

## Test fixtures

Real package directories live in `tests/test_cases/packages/`:

```
pkg_alpha/          prompts (flat + nested), 3 configs (default/dev/prod inheritance)
pkg_beta/           depends on pkg_alpha, own prompts + config
pkg_gamma/          custom sections: assets (JSON, PNG, YAML), data (JSON)
pkg_delta/          proxies (SearchProxy, AnalyticsProxy) + prompts
pkg_empty/          no resources — empty package
pkg_prefixed/       under-prefix path handling
pkg_cycle_a/        circular dependency with pkg_cycle_b
pkg_cycle_b/        circular dependency with pkg_cycle_a
pkg_chain_a/        4-level chain root (→ b → c → d)
pkg_chain_b/        4-level chain level 2
pkg_chain_c/        4-level chain level 3
pkg_chain_d/        4-level chain leaf
pkg_diamond_root/   diamond root (→ left + right)
pkg_diamond_left/   diamond left branch (→ shared)
pkg_diamond_right/  diamond right branch (→ shared)
pkg_diamond_shared/ diamond shared leaf
pkg_bad_toml/       intentionally malformed lllm.toml
```
