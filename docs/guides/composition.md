# Composition

Composition is ref resolution, not service launching.

A tactic can be called:

- in process,
- through an HTTP service,
- through a `psi://...` ref resolved from local config.

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

result = resolver.run("psi://demo/echo/tactics/echo", {"text": "hello"})
```

Local config can bind the same ref to a service:

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"
```

PsiHub validates and documents refs. It does not launch services.
