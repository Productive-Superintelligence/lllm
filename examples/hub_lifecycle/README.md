# Local Package Lifecycle Example

This example mirrors `docs/tutorials/local-package-lifecycle.md`.

It initializes a package, validates it, publishes it into a local hub, renders
cards/config, and downloads a clean copy.

```python
from pathlib import Path

from workflow import run_workflow

result = run_workflow(Path(".demo-lifecycle"))
print(result["record"].key)
```
