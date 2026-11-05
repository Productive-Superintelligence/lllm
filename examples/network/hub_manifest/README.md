# LLLM hub Manifest

This example builds channel and snapshot portions of a LLLM hub-style manifest
from LLLM network `Channel` and `Snapshot` objects.

```python
from examples.network.hub_manifest.package_manifest import build_manifest

manifest = build_manifest()
```

LLLM network does not publish packages itself. The generated structure is intended to
be copied into `lllm.toml` or passed to LLLM hub-side tooling that owns validation
and publication.
