import unittest
import tempfile
import textwrap
import warnings
from pathlib import Path

from pydantic import BaseModel


class EchoInput(BaseModel):
    text: str
    suffix: str = "!"


class EchoOutput(BaseModel):
    value: str


def _register_noop_invoker():
    from lllm.invokers import register_invoker

    register_invoker("noop", lambda config: object(), overwrite=True)


def _runtime_with_echo():
    from lllm.core.runtime import Runtime
    from lllm.core.tactic import Tactic, tactictool

    _register_noop_invoker()

    class EchoTactic(Tactic):
        """Echo text with a suffix."""

        name = "echo"
        agent_group = []

        @tactictool("echo", config={"invoker": "noop"})
        def call(self, task: EchoInput) -> EchoOutput:
            return EchoOutput(value=f"{task.text}{task.suffix}")

    rt = Runtime()
    rt.register_tactic("echo", EchoTactic, namespace="producer.tactics")
    rt.register_tactic("local_echo", EchoTactic, namespace="consumer.tactics")
    return rt, EchoTactic


def _runtime_with_regular_tool():
    from lllm.core.prompt import Function
    from lllm.core.runtime import Runtime

    def shout(text: str) -> str:
        return text.upper()

    tool_fn = Function.from_callable(
        shout,
        name="shout",
        description="Uppercase text.",
    )

    def local_shout(text: str) -> str:
        return text.upper()

    local_tool_fn = Function.from_callable(
        local_shout,
        name="local_shout",
        description="Uppercase text.",
    )

    rt = Runtime()
    rt.register_tool("shout", tool_fn, namespace="producer.tools")
    rt.register_tool("local_shout", local_tool_fn, namespace="consumer.tools")
    return rt, tool_fn


class TestTacticPromptToolRefs(unittest.TestCase):
    def test_tactictool_requires_explicit_name(self):
        from lllm.core.tactic import tactictool

        with self.assertRaisesRegex(ValueError, "explicit tool name|non-empty tool name"):
            tactictool()

    def test_prompt_function_list_resolves_full_tactic_url(self):
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tactics:echo"],
        )
        prompt._qualified_key = "consumer.prompts:main"

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved.functions)

        result = resolved.functions["echo"](
            FunctionCall(
                id="call_1",
                name="echo",
                arguments={"text": "hello", "suffix": "?"},
            )
        )
        self.assertEqual(result.result, '{"value":"hello?"}')

    def test_prompt_bare_tactic_ref_resolves_relative_to_prompt_package(self):
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["local_echo"],
        )
        prompt._qualified_key = "consumer.prompts:main"

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved.functions)

    def test_prompt_package_shorthand_resolves_as_tactic(self):
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer:echo"],
        )

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved.functions)

    def test_missing_config_raises_during_prompt_resolution(self):
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime
        from lllm.core.tactic import Tactic, tactictool

        _register_noop_invoker()

        class NoConfigTactic(Tactic):
            name = "no_config"
            agent_group = []

            @tactictool("no_config")
            def call(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text)

        rt = Runtime()
        rt.register_tactic("no_config", NoConfigTactic, namespace="producer.tactics")
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tactics:no_config"],
        )

        with self.assertRaisesRegex(ValueError, "no config binding"):
            prompt.resolve_function_refs(rt)

    def test_missing_description_and_schema_warns_with_fallback(self):
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime
        from lllm.core.tactic import Tactic, tactictool

        _register_noop_invoker()

        class FallbackTactic(Tactic):
            name = "fallback"
            agent_group = []

            @tactictool("fallback", config={"invoker": "noop"})
            def call(self, task):
                return f"done: {task}"

        rt = Runtime()
        rt.register_tactic("fallback", FallbackTactic, namespace="producer.tactics")
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tactics:fallback"],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = prompt.resolve_function_refs(rt)

        messages = [str(w.message) for w in caught]
        self.assertTrue(any("no description" in message for message in messages))
        self.assertTrue(any("unannotated parameter" in message for message in messages))
        self.assertEqual(resolved.functions["fallback"].required, ["task"])

    def test_url_fragment_selects_one_of_multiple_tactic_tools(self):
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime
        from lllm.core.tactic import Tactic, tactictool

        _register_noop_invoker()

        class MultiTactic(Tactic):
            name = "multi"
            agent_group = []

            def call(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text)

            @tactictool("upper", config={"invoker": "noop"})
            def upper_tool(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text.upper())

            @tactictool("lower", config={"invoker": "noop"})
            def lower_tool(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text.lower())

        rt = Runtime()
        rt.register_tactic("multi", MultiTactic, namespace="producer.tactics")
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tactics:multi#upper"],
        )

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("upper", resolved.functions)
        result = resolved.functions["upper"](
            FunctionCall(id="call_1", name="upper", arguments={"text": "Hello"})
        )
        self.assertEqual(result.result, '{"value":"HELLO"}')

    def test_multiple_tactic_tools_without_fragment_raises(self):
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime
        from lllm.core.tactic import Tactic, tactictool

        _register_noop_invoker()

        class MultiTactic(Tactic):
            name = "multi_ambiguous"
            agent_group = []

            def call(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text)

            @tactictool("first", config={"invoker": "noop"})
            def first(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text)

            @tactictool("second", config={"invoker": "noop"})
            def second(self, task: EchoInput) -> EchoOutput:
                return EchoOutput(value=task.text)

        rt = Runtime()
        rt.register_tactic("multi_ambiguous", MultiTactic, namespace="producer.tactics")
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tactics:multi_ambiguous"],
        )

        with self.assertRaisesRegex(ValueError, "multiple tactic tools"):
            prompt.resolve_function_refs(rt)


class TestRegisteredFunctionToolRefs(unittest.TestCase):
    def test_prompt_can_mix_direct_tool_url_and_declaration_styles(self):
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Function, Prompt

        rt, _ = _runtime_with_regular_tool()

        def count_chars(text: str) -> int:
            return len(text)

        direct_tool = Function.from_callable(
            count_chars,
            name="count_chars",
            description="Count characters.",
        )
        declaration = Function(
            name="local_shout",
            description="Uppercase text.",
            properties={"text": {"type": "string"}},
            required=["text"],
        )
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=[
                direct_tool,
                "producer.tools:shout",
                declaration,
            ],
        )
        prompt._qualified_key = "consumer.prompts:main"

        resolved = prompt.resolve_function_refs(rt)
        self.assertEqual(
            sorted(resolved.functions),
            ["count_chars", "local_shout", "shout"],
        )

        direct_result = resolved.functions["count_chars"](
            FunctionCall(id="call_1", name="count_chars", arguments={"text": "abc"})
        )
        url_result = resolved.functions["shout"](
            FunctionCall(id="call_2", name="shout", arguments={"text": "hello"})
        )
        declaration_result = resolved.functions["local_shout"](
            FunctionCall(id="call_3", name="local_shout", arguments={"text": "local"})
        )
        self.assertEqual(direct_result.result, 3)
        self.assertEqual(url_result.result, "HELLO")
        self.assertEqual(declaration_result.result, "LOCAL")

    def test_prompt_function_declaration_auto_binds_by_package_key(self):
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Function, Prompt

        rt, _ = _runtime_with_regular_tool()
        declaration = Function(
            name="shout",
            description="Uppercase text.",
            properties={"text": {"type": "string"}},
            required=["text"],
        )
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=[declaration],
        )
        prompt._qualified_key = "producer.prompts:main"

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("shout", resolved.functions)
        self.assertIsNotNone(resolved.functions["shout"].function)

        result = resolved.functions["shout"](
            FunctionCall(id="call_1", name="shout", arguments={"text": "hello"})
        )
        self.assertEqual(result.result, "HELLO")

    def test_prompt_function_declaration_requires_exact_tool_name_match(self):
        from lllm.core.prompt import Function, Prompt
        from lllm.core.resource import ResourceNode
        from lllm.core.runtime import Runtime

        _, tool_fn = _runtime_with_regular_tool()
        rt = Runtime()
        rt.register(ResourceNode.eager(
            "local_shout",
            tool_fn,
            namespace="consumer.tools",
            resource_type="tool",
        ))
        declaration = Function(
            name="local_shout",
            description="Uppercase text.",
            properties={"text": {"type": "string"}},
            required=["text"],
        )
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=[declaration],
        )
        prompt._qualified_key = "consumer.prompts:main"

        with self.assertRaisesRegex(ValueError, "exact name"):
            prompt.resolve_function_refs(rt)

    def test_prompt_function_list_resolves_registered_tool_url(self):
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_regular_tool()
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer.tools:shout"],
        )

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("shout", resolved.functions)

        result = resolved.functions["shout"](
            FunctionCall(id="call_1", name="shout", arguments={"text": "hello"})
        )
        self.assertEqual(result.result, "HELLO")

    def test_bare_registered_tool_ref_resolves_relative_to_prompt_package(self):
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_regular_tool()
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["local_shout"],
        )
        prompt._qualified_key = "consumer.prompts:main"

        resolved = prompt.resolve_function_refs(rt)
        self.assertIn("local_shout", resolved.functions)

    def test_ambiguous_tool_and_tactic_ref_requires_full_url(self):
        from lllm.core.prompt import Function
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()

        def echo(text: str) -> str:
            return text

        tool_fn = Function.from_callable(
            echo,
            name="echo",
            description="Echo text.",
        )
        rt.register_tool("echo", tool_fn, namespace="producer.tools")

        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["producer:echo"],
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            prompt.resolve_function_refs(rt)

    def test_tools_section_discovers_regular_tool_functions(self):
        from lllm.core.config import load_package
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "tools").mkdir()
            (tmp / "lllm.toml").write_text(textwrap.dedent("""\
                [package]
                name = "tool_pkg"

                [tools]
                paths = ["tools"]
            """))
            (tmp / "tools" / "basic.py").write_text(textwrap.dedent("""\
                from lllm import tool

                @tool(name="double", description="Double a number.")
                def double(value: int) -> int:
                    return value * 2
            """))
            (tmp / "tools" / "search.py").write_text(textwrap.dedent("""\
                from lllm import tool

                @tool(name="search", description="Search documents.")
                def search(query: str) -> str:
                    return f"result: {query}"
            """))

            rt = Runtime()
            load_package(str(tmp / "lllm.toml"), runtime=rt)

        self.assertTrue(rt.has("tool_pkg.tools:basic/double"))
        self.assertTrue(rt.has("tool_pkg.tools:search"))
        prompt = Prompt(
            path="main",
            prompt="Use tools.",
            function_list=["tool_pkg.tools:basic/double"],
        )
        resolved = prompt.resolve_function_refs(rt)
        result = resolved.functions["double"](
            FunctionCall(id="call_1", name="double", arguments={"value": 4})
        )
        self.assertEqual(result.result, 8)


class TestTacticProxyEndpoints(unittest.TestCase):
    def test_tactic_endpoint_appears_in_api_directory_and_dispatches(self):
        from lllm.proxies.base import BaseProxy, ProxyManager

        rt, _ = _runtime_with_echo()

        class SharedProxy(BaseProxy):
            _proxy_path = "shared"
            _proxy_name = "Shared Proxy"
            _proxy_description = "Shared tactic proxy"

            code_review = BaseProxy.tactic_endpoint(
                "producer.tactics:echo",
                endpoint="code_review",
            )

        rt.register_proxy("shared", SharedProxy, namespace="consumer.proxies")
        manager = ProxyManager(runtime=rt)

        directory = manager.get_api_directory("shared")
        endpoints = {entry["callable"]: entry for entry in directory["endpoints"]}
        self.assertIn("code_review", endpoints)
        self.assertIn("text*", endpoints["code_review"]["params"])

        result = manager("shared.code_review", {"text": "ok", "suffix": "."})
        self.assertEqual(result, '{"value":"ok."}')

    def test_proxy_register_tactic_bare_ref_resolves_relative_to_proxy_package(self):
        from lllm.proxies.base import BaseProxy, ProxyManager

        rt, _ = _runtime_with_echo()

        class LocalProxy(BaseProxy):
            _proxy_path = "local"
            _proxy_name = "Local Proxy"
            _proxy_description = "Local tactic proxy"

        LocalProxy.register_tactic(
            "local_echo",
            endpoint="run_local",
        )

        rt.register_proxy("local", LocalProxy, namespace="consumer.proxies")
        manager = ProxyManager(runtime=rt)

        result = manager("local.run_local", {"text": "local", "suffix": "!"})
        self.assertEqual(result, '{"value":"local!"}')


class TestAgentConfigTools(unittest.TestCase):
    def test_global_regular_tools_are_added_to_agent_prompt(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_regular_tool()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.tools:shout"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        resolved_prompt = agent.system_prompt.resolve_function_refs(rt)
        self.assertIn("shout", resolved_prompt.functions)
        result = resolved_prompt.functions["shout"](
            FunctionCall(id="call_1", name="shout", arguments={"text": "hello"})
        )
        self.assertEqual(result.result, "HELLO")

    def test_global_tools_are_added_to_agent_prompt(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.tactics:echo"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        resolved_prompt = agent.system_prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved_prompt.functions)

    def test_proxy_ref_in_tools_enables_proxy_programming(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime
        from lllm.proxies.base import BaseProxy

        class MarketProxy(BaseProxy):
            _proxy_path = "market"
            _proxy_name = "Market Proxy"
            _proxy_description = "Market data"

            @BaseProxy.endpoint(
                category="market",
                endpoint="quote",
                description="Get a quote.",
                params={"symbol*": (str, "AAPL")},
                response=["quote"],
            )
            def quote(self, symbol: str):
                return {"symbol": symbol, "price": 1}

        rt = Runtime()
        rt.register_proxy("market", MarketProxy, namespace="producer.proxies")
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.proxies:market"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        self.assertIn("query_api_doc", agent.system_prompt.functions)
        self.assertIn("run_python", agent.system_prompt.functions)
        self.assertIn("Market Proxy", agent.system_prompt.prompt)
        self.assertIn("quote", agent.system_prompt.prompt)

    def test_agent_config_can_mix_regular_tactic_and_proxy_refs(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.const import FunctionCall
        from lllm.core.prompt import Prompt
        from lllm.proxies.base import BaseProxy

        rt, _ = _runtime_with_echo()
        _, shout_fn = _runtime_with_regular_tool()
        rt.register_tool("shout", shout_fn, namespace="producer.tools")

        class MarketProxy(BaseProxy):
            _proxy_path = "market"
            _proxy_name = "Market Proxy"
            _proxy_description = "Market data"

            @BaseProxy.endpoint(
                category="market",
                endpoint="quote",
                description="Get a quote.",
                params={"symbol*": (str, "AAPL")},
                response=["quote"],
            )
            def quote(self, symbol: str):
                return {"symbol": symbol, "price": 1}

        rt.register_proxy("market", MarketProxy, namespace="producer.proxies")
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": [
                    "producer.tools:shout",
                    "producer.tactics:echo",
                    "producer.proxies:market",
                ],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        self.assertIn("query_api_doc", agent.system_prompt.functions)
        self.assertIn("run_python", agent.system_prompt.functions)

        resolved_prompt = agent.system_prompt.resolve_function_refs(rt)
        self.assertIn("shout", resolved_prompt.functions)
        self.assertIn("echo", resolved_prompt.functions)

        shout_result = resolved_prompt.functions["shout"](
            FunctionCall(id="call_1", name="shout", arguments={"text": "hello"})
        )
        echo_result = resolved_prompt.functions["echo"](
            FunctionCall(
                id="call_2",
                name="echo",
                arguments={"text": "hello", "suffix": "?"},
            )
        )
        self.assertEqual(shout_result.result, "HELLO")
        self.assertEqual(echo_result.result, '{"value":"hello?"}')

    def test_direct_tool_refs_are_lazy_until_prompt_resolution(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime

        rt = Runtime()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.tools:missing"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        self.assertIn("producer.tools:missing", agent.system_prompt.function_list)
        with self.assertRaisesRegex(KeyError, "Could not resolve tool resource"):
            agent.system_prompt.resolve_function_refs(rt)

    def test_proxy_refs_are_resolved_during_agent_build(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt
        from lllm.core.runtime import Runtime

        rt = Runtime()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.proxies:missing"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        with self.assertRaisesRegex(KeyError, "Could not resolve proxy resource"):
            spec.build(rt, object())

    def test_global_bare_tools_resolve_relative_to_prompt_package(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["local_echo"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        agent = spec.build(rt, object())

        resolved_prompt = agent.system_prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved_prompt.functions)

    def test_per_agent_tools_replace_global_tools(self):
        from lllm.core.config import parse_agent_configs
        from lllm.core.prompt import Prompt

        rt, _ = _runtime_with_echo()
        prompt = Prompt(path="system/main", prompt="Use tools.")
        rt.register_prompt(prompt, namespace="consumer.prompts")

        config = {
            "global": {
                "model_name": "noop-model",
                "tools": ["producer.tactics:missing"],
            },
            "agent_configs": [
                {
                    "name": "assistant",
                    "system_prompt_path": "consumer.prompts:system/main",
                    "tools": ["local_echo"],
                }
            ],
        }

        spec = parse_agent_configs(config, ["assistant"], "tool_test")["assistant"]
        self.assertEqual(spec.tools, ["local_echo"])
        agent = spec.build(rt, object())

        resolved_prompt = agent.system_prompt.resolve_function_refs(rt)
        self.assertIn("echo", resolved_prompt.functions)


if __name__ == "__main__":
    unittest.main()
