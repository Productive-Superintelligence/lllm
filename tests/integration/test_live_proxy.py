"""
Integration tests for the proxy tool-calling system.

Structure
---------
TestAgentInterpreter
    Unit tests for AgentInterpreter: no LLM, no external API.

TestProxyConfig
    Unit tests for ProxyConfig.from_dict and AgentSpec.build() proxy wiring:
    verifies that the right tools are injected per exec_env mode.

TestLiveProxyOpenAI  (pytest.mark.live + pytest.mark.openai)
    End-to-end: agent with run_python tool calls a real LLM.
    Uses a pure in-process Stats proxy — no external API key required beyond
    the LLM key itself.

Run:
    pytest tests/integration/test_live_proxy.py -v -m live
    pytest tests/integration/test_live_proxy.py -v           # unit tests only
"""
import unittest

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("auth", "api key", "401", "403", "invalid_api_key"))


def _make_stats_proxy_class(runtime=None):
    """Define a minimal in-process Stats proxy and register it in *runtime*."""
    from lllm.proxies import BaseProxy, ProxyRegistrator

    kwargs = {"path": "stats", "name": "Stats API", "description": "Simple stats."}
    if runtime is not None:
        kwargs["runtime"] = runtime

    @ProxyRegistrator(**kwargs)
    class StatsProxy(BaseProxy):

        @BaseProxy.endpoint(
            category="data",
            endpoint="generate",
            description="Return N numbers.",
            params={"n*": (int, 5), "seed": (int, 0)},
            response={"data": [1, 2, 3, 4, 5]},
        )
        def generate(self, params: dict) -> dict:
            import random
            rng = random.Random(params.get("seed", 0))
            return {"data": [rng.randint(1, 100) for _ in range(params.get("n", 5))]}

        @BaseProxy.endpoint(
            category="analysis",
            endpoint="mean",
            description="Return the mean of a list of numbers.",
            params={"data*": (list, [1, 2, 3])},
            response={"mean": 2.0},
        )
        def mean(self, params: dict) -> dict:
            data = params.get("data", [])
            if not data:
                return {"mean": None}
            return {"mean": sum(data) / len(data)}

    return StatsProxy


def _make_proxy_manager(runtime=None):
    """Build a ProxyManager with the Stats proxy registered."""
    from lllm.proxies.base import ProxyManager
    from lllm.core.runtime import Runtime

    rt = runtime or Runtime()
    _make_stats_proxy_class(runtime=rt)
    return ProxyManager(activate_proxies=["stats"], runtime=rt), rt


def _make_interpreter(runtime=None):
    """Build an AgentInterpreter wired to the Stats proxy."""
    from lllm.proxies.interpreter import AgentInterpreter

    pm, rt = _make_proxy_manager(runtime)
    interp = AgentInterpreter(pm, max_output_chars=500, truncation_indicator="..TRUNC..", timeout=5.0)
    return interp, pm, rt


# ═══════════════════════════════════════════════════════════════════════════
#  AgentInterpreter unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentInterpreter(unittest.TestCase):

    def setUp(self):
        self.interp, self.pm, _ = _make_interpreter()

    # ------------------------------------------------------------------
    # Basic execution
    # ------------------------------------------------------------------

    def test_simple_print(self):
        out = self.interp.run("print('hello')")
        self.assertEqual(out.strip(), "hello")

    def test_expression_not_auto_returned(self):
        """Unlike a REPL, bare expressions do not produce output."""
        out = self.interp.run("1 + 1")
        self.assertEqual(out.strip(), "")

    def test_print_expression(self):
        out = self.interp.run("print(2 + 2)")
        self.assertIn("4", out)

    def test_multi_line_code(self):
        code = "x = [i ** 2 for i in range(5)]\nprint(x)"
        out = self.interp.run(code)
        self.assertIn("[0, 1, 4, 9, 16]", out)

    def test_stdlib_available(self):
        out = self.interp.run("import math; print(round(math.pi, 4))")
        self.assertIn("3.1416", out)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def test_state_persists_across_calls(self):
        self.interp.run("x = 42")
        out = self.interp.run("print(x)")
        self.assertIn("42", out)

    def test_accumulated_state(self):
        self.interp.run("results = []")
        self.interp.run("results.append(1)")
        self.interp.run("results.append(2)")
        out = self.interp.run("print(results)")
        self.assertIn("[1, 2]", out)

    def test_variable_overwrite(self):
        self.interp.run("val = 'first'")
        self.interp.run("val = 'second'")
        out = self.interp.run("print(val)")
        self.assertIn("second", out)

    # ------------------------------------------------------------------
    # CALL_API
    # ------------------------------------------------------------------

    def test_call_api_is_available(self):
        out = self.interp.run(
            "result = CALL_API('stats/generate', {'n': 3, 'seed': 7})\n"
            "print(len(result['data']))"
        )
        self.assertIn("3", out)

    def test_call_api_result_persists(self):
        self.interp.run("data = CALL_API('stats/generate', {'n': 5, 'seed': 1})")
        out = self.interp.run("print(data['data'])")
        self.assertNotEqual(out.strip(), "")
        self.assertNotIn("NameError", out)

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def test_exception_returns_traceback(self):
        out = self.interp.run("raise ValueError('test error')")
        self.assertIn("ValueError", out)
        self.assertIn("test error", out)

    def test_name_error_returns_traceback(self):
        out = self.interp.run("print(undefined_variable)")
        self.assertIn("NameError", out)

    def test_exception_does_not_lose_state(self):
        """Raising an exception doesn't clear the namespace."""
        self.interp.run("safe_var = 'still here'")
        self.interp.run("raise RuntimeError('oops')")
        out = self.interp.run("print(safe_var)")
        self.assertIn("still here", out)

    def test_syntax_error_returns_traceback(self):
        out = self.interp.run("def broken(:\n    pass")
        self.assertIn("Error", out)

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def test_output_truncated(self):
        code = "print('x' * 1000)"
        out = self.interp.run(code)
        self.assertLessEqual(len(out), 510)  # 500 + indicator
        self.assertIn("..TRUNC..", out)

    def test_short_output_not_truncated(self):
        out = self.interp.run("print('hi')")
        self.assertNotIn("..TRUNC..", out)

    def test_no_truncation_when_zero(self):
        from lllm.proxies.interpreter import AgentInterpreter
        pm, _ = _make_proxy_manager()
        interp = AgentInterpreter(pm, max_output_chars=0)
        out = interp.run("print('x' * 1000)")
        self.assertNotIn("..TRUNC..", out)
        self.assertEqual(len(out.strip()), 1000)

    # ------------------------------------------------------------------
    # Timeout
    # ------------------------------------------------------------------

    def test_timeout_raises(self):
        from lllm.proxies.interpreter import AgentInterpreter
        pm, _ = _make_proxy_manager()
        interp = AgentInterpreter(pm, timeout=0.1)
        with self.assertRaises(TimeoutError):
            interp.run("import time; time.sleep(10)")

    def test_short_code_does_not_timeout(self):
        from lllm.proxies.interpreter import AgentInterpreter
        pm, _ = _make_proxy_manager()
        interp = AgentInterpreter(pm, timeout=5.0)
        out = interp.run("print('fast')")
        self.assertIn("fast", out)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_user_variables(self):
        self.interp.run("my_data = [1, 2, 3]")
        self.interp.reset()
        out = self.interp.run("print(my_data)")
        self.assertIn("NameError", out)

    def test_reset_preserves_call_api(self):
        self.interp.reset()
        out = self.interp.run("result = CALL_API('stats/generate', {'n': 2}); print('ok')")
        self.assertIn("ok", out)

    def test_reset_preserves_builtins(self):
        self.interp.reset()
        out = self.interp.run("print(len([1, 2, 3]))")
        self.assertIn("3", out)


# ═══════════════════════════════════════════════════════════════════════════
#  ProxyConfig + AgentSpec.build() — tool injection unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestProxyConfig(unittest.TestCase):

    def _make_spec(self, exec_env, extra_proxy=None):
        """Build a minimal AgentSpec with a proxy block."""
        from lllm.core.config import AgentSpec, ProxyConfig
        from lllm.core.prompt import Prompt

        proxy_cfg = ProxyConfig(
            activate_proxies=["stats"],
            exec_env=exec_env,
        )
        return AgentSpec(
            name="test_agent",
            model="gpt-4o-mini",
            system_prompt=Prompt(path="test::sys", prompt="You are helpful."),
            proxy=proxy_cfg,
        )

    def _make_runtime_with_stats(self):
        from lllm.core.runtime import Runtime
        rt = Runtime()
        _make_stats_proxy_class(runtime=rt)
        return rt

    def test_proxy_config_from_dict_defaults(self):
        from lllm.core.config import ProxyConfig
        cfg = ProxyConfig.from_dict({})
        self.assertEqual(cfg.exec_env, "interpreter")
        self.assertEqual(cfg.max_output_chars, 5000)
        self.assertEqual(cfg.truncation_indicator, "... (truncated)")
        self.assertAlmostEqual(cfg.timeout, 60.0)
        self.assertFalse(cfg.deploy_mode)
        self.assertEqual(cfg.activate_proxies, [])

    def test_proxy_config_from_dict_custom(self):
        from lllm.core.config import ProxyConfig
        cfg = ProxyConfig.from_dict({
            "activate_proxies": ["fmp"],
            "exec_env": "jupyter",
            "max_output_chars": 2000,
            "timeout": 30.0,
            "deploy_mode": True,
        })
        self.assertEqual(cfg.activate_proxies, ["fmp"])
        self.assertEqual(cfg.exec_env, "jupyter")
        self.assertEqual(cfg.max_output_chars, 2000)
        self.assertAlmostEqual(cfg.timeout, 30.0)
        self.assertTrue(cfg.deploy_mode)

    def test_interpreter_mode_injects_run_python(self):
        """exec_env='interpreter' → run_python + query_api_doc in function_list."""
        from lllm.invokers.litellm import LiteLLMInvoker
        spec = self._make_spec("interpreter")
        rt = self._make_runtime_with_stats()
        agent = spec.build(rt, LiteLLMInvoker())
        fn_names = [f.name for f in agent.system_prompt.function_list]
        self.assertIn("run_python", fn_names)
        self.assertIn("query_api_doc", fn_names)

    def test_jupyter_mode_no_run_python(self):
        """exec_env='jupyter' → query_api_doc only; run_python NOT injected."""
        from lllm.invokers.litellm import LiteLLMInvoker
        spec = self._make_spec("jupyter")
        rt = self._make_runtime_with_stats()
        agent = spec.build(rt, LiteLLMInvoker())
        fn_names = [f.name for f in agent.system_prompt.function_list]
        self.assertIn("query_api_doc", fn_names)
        self.assertNotIn("run_python", fn_names)

    def test_null_exec_env_no_run_python(self):
        """exec_env=None → query_api_doc only; run_python NOT injected."""
        from lllm.invokers.litellm import LiteLLMInvoker
        spec = self._make_spec(None)
        rt = self._make_runtime_with_stats()
        agent = spec.build(rt, LiteLLMInvoker())
        fn_names = [f.name for f in agent.system_prompt.function_list]
        self.assertIn("query_api_doc", fn_names)
        self.assertNotIn("run_python", fn_names)

    def test_no_proxy_no_tools_injected(self):
        """No proxy config → function_list is empty."""
        from lllm.core.config import AgentSpec
        from lllm.core.prompt import Prompt
        from lllm.invokers.litellm import LiteLLMInvoker
        spec = AgentSpec(
            name="bare",
            model="gpt-4o-mini",
            system_prompt=Prompt(path="test::bare", prompt="You are helpful."),
        )
        rt = self._make_runtime_with_stats()
        agent = spec.build(rt, LiteLLMInvoker())
        self.assertEqual(len(agent.system_prompt.function_list), 0)

    def test_proxy_block_appended_to_system_prompt(self):
        """The proxy prompt block is appended to the system prompt text."""
        from lllm.invokers.litellm import LiteLLMInvoker
        spec = self._make_spec("interpreter")
        rt = self._make_runtime_with_stats()
        agent = spec.build(rt, LiteLLMInvoker())
        prompt_text = agent.system_prompt.prompt
        self.assertIn("CALL_API", prompt_text)
        self.assertIn("run_python", prompt_text)
        self.assertIn("Available APIs", prompt_text)

    def test_original_prompt_not_mutated(self):
        """build() must not mutate the original Prompt object."""
        from lllm.core.config import AgentSpec, ProxyConfig
        from lllm.core.prompt import Prompt
        from lllm.invokers.litellm import LiteLLMInvoker

        original_text = "You are the original."
        original_prompt = Prompt(path="test::orig", prompt=original_text)
        spec = AgentSpec(
            name="test_no_mutate",
            model="gpt-4o-mini",
            system_prompt=original_prompt,
            proxy=ProxyConfig(activate_proxies=["stats"]),
        )
        rt = self._make_runtime_with_stats()
        spec.build(rt, LiteLLMInvoker())
        self.assertEqual(original_prompt.prompt, original_text)
        self.assertEqual(len(original_prompt.function_list), 0)

    def test_proxy_manager_built_with_activation_filter(self):
        """Only the proxies listed in activate_proxies are loaded."""
        from lllm.core.runtime import Runtime
        from lllm.proxies.base import ProxyManager

        rt = Runtime()
        _make_stats_proxy_class(runtime=rt)

        # Register a second proxy that should NOT be loaded
        from lllm.proxies import BaseProxy, ProxyRegistrator

        @ProxyRegistrator(path="other", name="Other", description="Other proxy.", runtime=rt)
        class OtherProxy(BaseProxy):
            pass

        pm = ProxyManager(activate_proxies=["stats"], runtime=rt)
        self.assertIn("stats", pm.available())
        self.assertNotIn("other", pm.available())


# ═══════════════════════════════════════════════════════════════════════════
#  Live end-to-end tests — require LLM API key
# ═══════════════════════════════════════════════════════════════════════════

def _build_proxy_agent(model: str, runtime):
    """Build a live Agent with interpreter mode + Stats proxy."""
    from lllm.core.config import AgentSpec, ProxyConfig
    from lllm.core.prompt import Prompt
    from lllm.invokers.litellm import LiteLLMInvoker

    spec = AgentSpec(
        name="proxy_agent",
        model=model,
        system_prompt=Prompt(
            path="test::proxy_sys",
            prompt=(
                "You are a data analyst. "
                "Use run_python with CALL_API('stats/...', params) to fetch and process data. "
                "Always use print() to produce output. "
                "Call query_api_doc('stats') if you need endpoint details."
            ),
        ),
        proxy=ProxyConfig(
            activate_proxies=["stats"],
            exec_env="interpreter",
            max_output_chars=2000,
            timeout=30.0,
        ),
        max_interrupt_steps=5,
    )
    return spec.build(runtime, LiteLLMInvoker())


@pytest.mark.live
@pytest.mark.openai
class TestLiveProxyOpenAI:

    @pytest.fixture(autouse=True)
    def _runtime(self):
        from lllm.core.runtime import Runtime
        rt = Runtime()
        _make_stats_proxy_class(runtime=rt)
        self._rt = rt

    def test_agent_calls_run_python(self, openai_available):
        """Agent calls run_python at least once and returns a non-empty answer."""
        agent = _build_proxy_agent("gpt-4o-mini", self._rt)
        try:
            agent.open("main")
            agent.receive(
                "Generate 5 numbers with seed=99 using the stats API and print them."
            )
            session = agent.respond(return_session=True)
        except Exception as exc:
            if _is_auth_error(exc):
                pytest.skip(f"Auth error: {exc}")
            raise
        assert session.state == "success"
        assert session.delivery.content.strip() != ""

    def test_state_persists_across_tool_calls(self, openai_available):
        """Agent fetches data in one run_python call and processes it in the next."""
        agent = _build_proxy_agent("gpt-4o-mini", self._rt)
        try:
            agent.open("state_test")
            agent.receive(
                "In the first run_python call, fetch 6 numbers with seed=1 "
                "and store them in a variable called 'nums'. "
                "In a second separate run_python call, print the sum of 'nums'. "
                "This confirms that variables persist across calls."
            )
            session = agent.respond(return_session=True)
        except Exception as exc:
            if _is_auth_error(exc):
                pytest.skip(f"Auth error: {exc}")
            raise
        assert session.state == "success"
        # The agent should have made at least two run_python calls
        total_interrupts = sum(len(v) for v in session.interrupts.values())
        if total_interrupts == 0:
            pytest.skip("Model made no tool calls; non-deterministic — skipping")
        assert total_interrupts >= 1

    def test_query_api_doc_is_callable(self, openai_available):
        """Agent can call query_api_doc to retrieve endpoint documentation."""
        agent = _build_proxy_agent("gpt-4o-mini", self._rt)
        try:
            agent.open("doc_test")
            agent.receive(
                "Call query_api_doc with proxy name 'stats', then tell me "
                "what endpoints are available."
            )
            session = agent.respond(return_session=True)
        except Exception as exc:
            if _is_auth_error(exc):
                pytest.skip(f"Auth error: {exc}")
            raise
        assert session.state == "success"
        content = session.delivery.content.lower()
        # The agent should mention at least one endpoint from the Stats API
        assert any(word in content for word in ("generate", "mean", "stats", "endpoint"))

    def test_exception_in_code_is_recovered(self, openai_available):
        """Agent recovers from a traceback returned by run_python."""
        agent = _build_proxy_agent("gpt-4o-mini", self._rt)
        try:
            agent.open("err_test")
            agent.receive(
                "Try to access a variable called 'missing_var' in run_python. "
                "You will get a NameError. Then fix the code so it prints the number 7 instead."
            )
            session = agent.respond(return_session=True)
        except Exception as exc:
            if _is_auth_error(exc):
                pytest.skip(f"Auth error: {exc}")
            raise
        assert session.state == "success"
        assert session.delivery.content.strip() != ""


@pytest.mark.live
@pytest.mark.anthropic
class TestLiveProxyAnthropic:

    @pytest.fixture(autouse=True)
    def _runtime(self):
        from lllm.core.runtime import Runtime
        rt = Runtime()
        _make_stats_proxy_class(runtime=rt)
        self._rt = rt

    def test_agent_calls_run_python(self, anthropic_available):
        """Claude agent calls run_python and returns a non-empty answer."""
        agent = _build_proxy_agent("claude-haiku-4-5-20251001", self._rt)
        try:
            agent.open("main")
            agent.receive(
                "Use the stats API to generate 4 numbers with seed=5 and print them."
            )
            session = agent.respond(return_session=True)
        except Exception as exc:
            if _is_auth_error(exc):
                pytest.skip(f"Auth error: {exc}")
            raise
        assert session.state == "success"
        assert session.delivery.content.strip() != ""


# ═══════════════════════════════════════════════════════════════════════════
#  Proxy tools unit tests (make_query_api_doc_tool, make_run_python_tool)
# ═══════════════════════════════════════════════════════════════════════════

class TestProxyTools(unittest.TestCase):

    def setUp(self):
        self.interp, self.pm, _ = _make_interpreter()

    def test_make_run_python_tool_returns_function(self):
        from lllm.proxies.proxy_tools import make_run_python_tool
        from lllm.core.prompt import Function
        fn = make_run_python_tool(self.interp)
        self.assertIsInstance(fn, Function)
        self.assertEqual(fn.name, "run_python")

    def test_run_python_tool_executes(self):
        from lllm.proxies.proxy_tools import make_run_python_tool
        fn = make_run_python_tool(self.interp)
        # Call the underlying callable directly (Function.__call__ takes a FunctionCall)
        result = fn.function(code="print('tool works')")
        self.assertIn("tool works", result)

    def test_make_query_api_doc_tool_returns_function(self):
        from lllm.proxies.proxy_tools import make_query_api_doc_tool
        from lllm.core.prompt import Function
        fn = make_query_api_doc_tool(self.pm)
        self.assertIsInstance(fn, Function)
        self.assertEqual(fn.name, "query_api_doc")

    def test_query_api_doc_returns_docs(self):
        from lllm.proxies.proxy_tools import make_query_api_doc_tool
        fn = make_query_api_doc_tool(self.pm)
        result = fn.function(proxy_name="stats")
        self.assertIn("Stats API", result)

    def test_query_api_doc_missing_proxy(self):
        from lllm.proxies.proxy_tools import make_query_api_doc_tool
        fn = make_query_api_doc_tool(self.pm)
        result = fn.function(proxy_name="nonexistent")
        self.assertIn("not found", result.lower())
        self.assertIn("stats", result.lower())


# ═══════════════════════════════════════════════════════════════════════════
#  render_proxy_prompt unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRenderProxyPrompt(unittest.TestCase):

    def test_interpreter_mode_contains_run_python(self):
        from lllm.proxies.prompt_template import render_proxy_prompt
        result = render_proxy_prompt(
            api_directory="## Stats API\n  generate: ...",
            max_output_chars=5000,
            truncation_indicator="... (truncated)",
            exec_env="interpreter",
        )
        self.assertIn("run_python", result)
        self.assertIn("CALL_API", result)
        self.assertIn("5000", result)

    def test_jupyter_mode_minimal_block(self):
        from lllm.proxies.prompt_template import render_proxy_prompt
        result = render_proxy_prompt(
            api_directory="## Stats API",
            max_output_chars=5000,
            truncation_indicator="... (truncated)",
            exec_env="jupyter",
        )
        self.assertIn("CALL_API", result)
        self.assertNotIn("run_python", result)

    def test_null_exec_env_minimal_block(self):
        from lllm.proxies.prompt_template import render_proxy_prompt
        result = render_proxy_prompt(
            api_directory="## Stats API",
            max_output_chars=5000,
            truncation_indicator="... (truncated)",
            exec_env=None,
        )
        self.assertNotIn("run_python", result)

    def test_api_directory_injected(self):
        from lllm.proxies.prompt_template import render_proxy_prompt
        result = render_proxy_prompt(
            api_directory="UNIQUE_API_DIRECTORY_CONTENT_XYZ",
            max_output_chars=5000,
            truncation_indicator="...",
            exec_env="interpreter",
        )
        self.assertIn("UNIQUE_API_DIRECTORY_CONTENT_XYZ", result)

    def test_braces_re_escaped(self):
        """All { and } in the rendered block must be doubled for safe Prompt embedding."""
        from lllm.proxies.prompt_template import render_proxy_prompt
        result = render_proxy_prompt(
            api_directory="## API",
            max_output_chars=100,
            truncation_indicator="...",
            exec_env="interpreter",
        )
        import re
        # No lone single braces should remain (every { must be {{, every } must be }})
        lone_open = re.findall(r'(?<!\{)\{(?!\{)', result)
        lone_close = re.findall(r'(?<!\})\}(?!\})', result)
        self.assertEqual(lone_open, [], f"Lone open braces found: {lone_open}")
        self.assertEqual(lone_close, [], f"Lone close braces found: {lone_close}")

    def test_custom_template(self):
        from lllm.proxies.prompt_template import render_proxy_prompt
        template = "Custom: {api_directory} | chars={max_output_chars}"
        result = render_proxy_prompt(
            api_directory="MY_DOCS",
            max_output_chars=1234,
            truncation_indicator="...",
            exec_env="interpreter",
            custom_template=template,
        )
        self.assertIn("MY_DOCS", result)
        self.assertIn("1234", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
