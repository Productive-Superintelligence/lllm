"""Native package loading: custom sections, services, and typed lookups."""

import textwrap

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.core.config import load_package


def write(root, files):
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def test_python_files_in_custom_sections_and_services_load(tmp_path):
    write(
        tmp_path,
        {
            "lllm.toml": """
                [package]
                name = "demo"

                [extras]
                paths = ["./extras"]

                [services]
                paths = ["./services"]
            """,
            "extras/helpers.py": "VALUE = 1\n",
            "services/app.py": """
                from fastapi import FastAPI

                app = FastAPI()
            """,
            "services/api.yaml": "port: 8080\n",
        },
    )
    runtime = Runtime()

    load_package(str(tmp_path / "lllm.toml"), runtime=runtime)

    assert runtime.get_service("demo.services:app/app").routes is not None
    assert runtime.get_service("demo.services:api.yaml") == {"port": 8080}


def test_typed_lookups_use_the_plural_section_name(tmp_path):
    write(
        tmp_path,
        {
            "lllm.toml": """
                [package]
                name = "demo"

                [proxies]
                paths = ["./proxies"]
            """,
            "proxies/weather.py": """
                from lllm.runtimes.native.proxies import BaseProxy


                class WeatherProxy(BaseProxy):
                    pass
            """,
        },
    )
    runtime = Runtime()
    load_package(str(tmp_path / "lllm.toml"), runtime=runtime)
    key = next(k for k in runtime.keys("proxy"))
    short = "demo:" + key.split(":", 1)[1]

    assert runtime.get(short, resource_type="proxy").__name__ == "WeatherProxy"


def test_prompts_without_variables_unescape_braces_and_use_custom_renderers():
    from lllm.runtimes.native import Prompt
    from lllm.runtimes.native.core.prompt import BaseRenderer

    class Shouting(BaseRenderer):
        def render(self, prompt, **kwargs):
            return prompt.format(**kwargs).upper()

    assert Prompt(path="json", prompt='Reply as {{"ok": true}}.')() == 'Reply as {"ok": true}.'
    assert Prompt(path="loud", prompt="hello", renderer=Shouting())() == "HELLO"


class Verdict(__import__("pydantic").BaseModel):
    ok: bool


def test_dialogs_with_tools_and_formats_serialize_to_json():
    import json

    from lllm.runtimes.native import Dialog, Prompt, tool

    @tool(description="Echo")
    def echo(text: str) -> str:
        return text

    dialog = Dialog(owner="a")
    dialog.put_prompt(
        Prompt(path="p", prompt="Judge {x}.", function_list=[echo], format=Verdict),
        {"x": "it"},
    )

    data = json.loads(json.dumps(dialog.to_dict()))
    restored = Dialog.from_dict(data)

    assert data["top_prompt"]["function_list"][0]["name"] == "echo"
    assert restored.top_prompt.format is Verdict
    assert restored.top_prompt.functions["echo"].function is None


def test_custom_renderers_receive_variables_without_format_fields():
    import string

    from lllm.runtimes.native import Prompt
    from lllm.runtimes.native.core.prompt import BaseRenderer

    class TemplateRenderer(BaseRenderer):
        def render(self, prompt, **kwargs):
            return string.Template(prompt).substitute(**kwargs)

    prompt = Prompt(path="dollar", prompt="Hello $name.", renderer=TemplateRenderer())

    assert prompt(name="Ada") == "Hello Ada."
