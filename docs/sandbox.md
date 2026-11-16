# Sandbox

The sandbox gives an agent a real Python kernel to work in. Each session is
a Jupyter kernel with a notebook file: the agent writes cells, your tactic
runs them, and every cell and its output is kept in an ordinary `.ipynb`.
Unlike `run_python`, which runs code inside your process, the kernel is a
separate process, it can draw plots, and the notebook is a record of the
work you can open later. [Proxies](proxies.md) work there too: a setup
cell defines `CALL_API` in the kernel.

Here a session runs one cell against the weather proxy from
[Proxies](proxies.md), saved as `proxies/weather.py` in a project whose
`lllm.toml` lists that folder
([Proxies in packages](proxies.md#proxies-in-packages)):

```python title="first_session.py" test="skip" reason="needs a Jupyter kernel"
from pathlib import Path

from lllm.runtimes.native.sandbox import JupyterSandbox

sandbox = JupyterSandbox({
    "name": "weather",
    "project_root": str(Path.cwd()),  # (1)!
    "activate_proxies": ["weather"],
})
session = sandbox.new_session("first")
print(session.run_cell(0))  # (2)!
#> True

code = """
spread = {}
for city in ["Tokyo", "Paris"]:
    day = CALL_API(f"weather/forecast/{city}")
    spread[city] = day["high"] - day["low"]
print(spread)
max(spread, key=spread.get)
"""
index = session.append_code_cell(code.strip())
print(index, session.run_cell(index))
#> 1 True
stream, result = session.cells[index].outputs  # (3)!
print(stream.output_type, repr(stream.text))
#> stream "{'Tokyo': 7, 'Paris': 8}\n"
print(result.output_type, result.data["text/plain"])
#> execute_result 'Paris'
print(session.notebook_file)
#> /home/you/.lllm/sandbox/weather/sessions/first/first.ipynb
session.shutdown()
```

1. The project folder: the kernel adds it to `sys.path` and loads its
   `lllm.toml`, so the project's proxies are available there.
2. Cell 0 is a setup cell LLLM writes into every notebook. It defines
   `CALL_API`; run it before the agent's cells.
3. Outputs come back as notebook outputs: printed text, the value of the
   last expression, and errors, the same as in Jupyter.

## Setup

```bash
pip install "lllm-core[native,sandbox]"
```

The `sandbox` extra installs `nbformat`, `jupyter-client`, and `ipykernel`,
which provides the Python kernel. Install it in the environment that has
LLLM and your project, since the kernel imports both. Without a kernel,
`run_cell()` returns `False` and logs `No such kernel named python3`.

## Sessions and notebooks

`JupyterSandbox(config, path=None)` manages a folder of sessions. Its config
takes:

| Key | Meaning |
| --- | --- |
| `name` | The sandbox's folder name |
| `project_root` | A folder with `lllm.toml`; the kernel puts it on `sys.path` and loads its proxies |
| `activate_proxies` | The proxies `CALL_API` reaches, for new sessions |
| `autorun_sessions` | Optional: run the notebook as soon as a session is created or reopened |

Sessions live in `~/.lllm/sandbox/<name>/sessions/<session>/` as
`<session>.ipynb` and `<session>_meta.json`. If the `TMP_DIR` environment
variable is set, it replaces `~/.lllm`, and a `path` replaces
`~/.lllm/sandbox/<name>`. `new_session(name=None, metadata=None)` creates a
session, named by the date and time if you give no name;
`metadata={"proxy": {...}}` can set `activate_proxies`, `cutoff_date`, or
`deploy_mode` for that session.
`get_session(name)` returns an open session, reopens one from its folder
with its cells and outputs, or creates it.

The setup cell in cell 0 adds `project_root` to `sys.path`, loads the
project's `lllm.toml` if there is one, registers the
[built-in proxies](proxies.md#built-in-proxies), and builds `CALL_API` for
the session's proxies. It is written when the session is created, not run:
run it yourself, or set `autorun_sessions`.

| `JupyterSession` | Does |
| --- | --- |
| `append_code_cell(source)`, `append_markdown_cell(text)` | Add a cell; returns its index |
| `run_cell(index, timeout=60)` | Run a code cell; `True` if it succeeded. Starts the kernel on first use |
| `run_all_cells(stop_on_error=True)` | Run the notebook; returns the index of the first failed cell, or `None` |
| `cells`, `get_cells(indices)`, `n_cells` | The notebook's cells, read from the file, with their outputs |
| `insert_cell`, `overwrite_cell`, `delete_cells` | Edit the notebook |
| `shutdown_kernel()`, `shutdown()` | Stop the kernel (and everything the session started) |

A cell's outputs are `stream` outputs (`name`, `text`), `execute_result` and
`display_data` outputs (`data` by MIME type, such as `text/plain`), and
`error` outputs (`ename`, `evalue`, and a colored `traceback`). A plot
arrives as `display_data` with `image/png` data, a base64 PNG that
`agent.receive_image()` can show to a model that reads images.

## Letting an agent write cells

The agent writes cells as tags in its reply, and a parser on the turn's
prompt reads them, so a reply without cells is repaired before your code
sees it:

```python title="notebook_turn.py"
import re

from lllm import ParseError, Prompt

SYSTEM = """You are a data analyst working in a Jupyter notebook.
Write code in <python_cell></python_cell> tags and notes in
<markdown_cell></markdown_cell> tags. Cells run in order and keep their
variables. Print what you need to see. CALL_API(path, params) calls the
APIs listed below. When you know the answer, reply with it in
<answer></answer> tags."""

CELL = re.compile(r"<(python_cell|markdown_cell)>(.*?)</\1>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def read_turn(text: str, **kwargs) -> dict:
    cells = [(kind, body.strip()) for kind, body in CELL.findall(text)]
    answer = ANSWER.search(text)
    if not cells and not answer:
        raise ParseError("Write a <python_cell>, or give your <answer>.")
    return {"cells": cells, "answer": answer and answer.group(1).strip()}


TURN = Prompt(path="notebook/turn", prompt="{content}", parser=read_turn)

print(TURN.parse("<python_cell>print(6 * 7)</python_cell>")["cells"])
#> [('python_cell', 'print(6 * 7)')]
try:
    TURN.parse("Let me think about it.")
except ParseError as error:
    print(error)
    #> Write a <python_cell>, or give your <answer>.
```

The tactic runs each code cell and sends the outputs back as the next turn,
until the agent answers. The agent's config uses `exec_env: jupyter`, which
puts the API directory in its system prompt and gives it `query_api_doc`,
but no `run_python`, since your tactic runs the code:

```python title="notebook_analyst.py" test="skip" reason="needs a Jupyter kernel"
from pathlib import Path

from lllm import Tactic
from lllm.runtimes.native.sandbox import JupyterSandbox
from notebook_turn import SYSTEM, TURN


def cell_text(cell) -> str:
    parts = []
    for out in cell.outputs:
        if out.output_type == "stream":
            parts.append(out.text)
        elif out.output_type in ("execute_result", "display_data"):
            parts.append(out.data.get("text/plain", "[image]") + "\n")
        elif out.output_type == "error":
            parts.append(f"{out.ename}: {out.evalue}\n")  # (1)!
    return "".join(parts)


class NotebookAnalyst(Tactic):
    name = "notebook_analyst"
    agent_group = ["analyst"]

    def call(self, question: str) -> dict:
        sandbox = JupyterSandbox({
            "name": "analyst",
            "project_root": str(Path.cwd()),
            "activate_proxies": ["weather"],
        })
        session = sandbox.new_session()
        session.run_cell(0)
        analyst = self.agents["analyst"]
        analyst.open("work")
        content = question
        try:
            for _ in range(8):
                analyst.receive_prompt(TURN, {"content": content})
                turn = analyst.respond().parsed
                if turn["answer"]:
                    notebook = session.notebook_file
                    return {"answer": turn["answer"], "notebook": notebook}
                results = []
                for kind, source in turn["cells"]:
                    if kind == "markdown_cell":
                        session.append_markdown_cell(source)
                        continue
                    index = session.append_code_cell(source)
                    status = "ok" if session.run_cell(index) else "error"
                    text = cell_text(session.cells[index])
                    results.append(f"[cell {index}: {status}]\n{text}")
                content = "\n".join(results) or "No code ran."
        finally:
            session.shutdown()  # (2)!
        raise RuntimeError("No answer after 8 turns.")


analyst = NotebookAnalyst({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [{
        "name": "analyst",
        "system_prompt": SYSTEM,
        "proxy": {"activate_proxies": ["weather"], "exec_env": "jupyter"},
    }],
})
result = analyst("Which city has the wider temperature range tomorrow?")
print(result["answer"])
#> Paris: 8°C between low and high (11 to 19°C), against 7°C in Tokyo.
```

1. Errors go back as one line. The full traceback is in the notebook, with
   terminal colors.
2. The kernel stops with the call; the notebook stays on disk.

In this run the agent's first cell had a typo. These are the two results it
received, and the second cell fixed the mistake using `days`, which the
first cell had already fetched:

```text
[cell 2: error]
NameError: name 'day' is not defined

[cell 3: ok]
{'Tokyo': 7, 'Paris': 8}
```

## Limits

- **Time.** A cell that runs past `run_cell`'s `timeout` (60 seconds by
  default) gets a `TimeoutError` output and `run_cell` returns `False`, but
  the kernel keeps running it, and later cells wait behind it. Call
  `session.shutdown_kernel()`: the next `run_cell` starts a fresh kernel,
  without the old variables, so run cell 0 again.
- **Output.** Outputs are stored in full. Cut long text before you send it
  to the model, as `run_python` does with `max_output_chars`.
- **Kernels.** Each session has its own kernel, which starts on first use
  and waits up to 10 seconds to be ready; kernels start one at a time. Shut
  sessions down when you are done (`sandbox.shutdown_all_sessions_resources()`
  stops every session the sandbox opened).

## Safety

The kernel is a separate process, but not a security boundary: it runs as
your user, with your files and network, and it inherits your environment
variables, API keys included. The in-process
[interpreter](proxies.md#the-interpreter) behind `run_python` isolates even
less. Both suit trusted agents working on your own data and APIs.

Because your tactic runs every cell, it decides what runs where: it can
refuse a cell, log it, or ask a person first. For code you do not trust,
run cells in a container or virtual machine, or on a remote execution
service; only the line that runs a cell changes. This runner gives each
cell a fresh container with no network:

```python title="container_runner.py" test="skip" reason="needs Docker"
import subprocess


def run_in_container(code: str, timeout: float = 60) -> str:
    command = [
        "docker", "run", "--rm", "--network", "none", "--memory", "512m",
        "python:3.12-slim", "python", "-c", code,
    ]
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout
    )
    return done.stdout + done.stderr


print(run_in_container("print(sum(range(10)))"))
#> 45
```

Cells there share no variables and cannot reach `CALL_API`, so ask the
agent for self-contained cells, and have your tactic fetch the data and
pass it in.

## Limits for a whole tactic

`sandbox_tactic()` is a different kind of limit: it wraps a
[tactic](tactics.md) at its typed boundary and checks each call against a
`SandboxPolicy`. It does not isolate code; it bounds what a served tactic
accepts and returns:

```python title="policy.py"
from lllm import SandboxLimitError, Tactic, sandbox_tactic


class Echo(Tactic):
    name = "echo"

    def call(self, text: str) -> str:
        return text


echo = sandbox_tactic(Echo(), {"max_input_bytes": 64})
print(echo.run("short"))
#> short
try:
    echo.run("x" * 100)
except SandboxLimitError as error:
    print(error)
    #> input is 102 bytes; limit is 64 bytes.
```

| `SandboxPolicy` field | Rejects |
| --- | --- |
| `max_input_bytes`, `max_output_bytes` | Inputs or outputs larger than this, as JSON (`SandboxLimitError`) |
| `allowed_metadata_keys` | Calls whose `CallContext.metadata` has other keys (`SandboxLimitError`) |
| `timeout_seconds` | Async calls (`arun()`, and served requests) that take longer (`SandboxTimeoutError`) |
