"""
Tests for lllm/core/prompt.py
Covers: Prompt, Function, @tool decorator, DefaultTagParser, BaseParser,
        StringFormatterRenderer, DefaultSimpleHandler, AgentCallSession, MCP.
"""
import unittest
from typing import Optional
from unittest.mock import MagicMock

# prompt.py uses `from __future__ import annotations` which makes ALL annotations
# lazy strings.  Pydantic 2 needs model_rebuild() after all referenced types load.
from lllm.core.dialog import Message  # noqa: F401 — must be imported first
from lllm.core.prompt import AgentCallSession
AgentCallSession.model_rebuild(force=True)


# ===========================================================================
# StringFormatterRenderer
# ===========================================================================


class TestStringFormatterRenderer(unittest.TestCase):

    def setUp(self):
        from lllm.core.prompt import StringFormatterRenderer
        self.renderer = StringFormatterRenderer()

    def test_no_kwargs_returns_prompt_unchanged(self):
        result = self.renderer.render("Hello world")
        self.assertEqual(result, "Hello world")

    def test_single_var(self):
        result = self.renderer.render("Hello {name}!", name="Alice")
        self.assertEqual(result, "Hello Alice!")

    def test_multiple_vars(self):
        result = self.renderer.render("{a} + {b} = {c}", a="1", b="2", c="3")
        self.assertEqual(result, "1 + 2 = 3")

    def test_escaped_braces_not_replaced(self):
        result = self.renderer.render("Use {{double}} for literal braces", foo="bar")
        self.assertEqual(result, "Use {double} for literal braces")


# ===========================================================================
# DefaultTagParser
# ===========================================================================


class TestDefaultTagParser(unittest.TestCase):
    """
    NOTE: DefaultTagParser raises ParseError if ANY xml_tags or md_tags entry
    has zero matches — all listed tags are required.  Each test uses an
    appropriately scoped parser so the content always satisfies all requirements.
    """

    def test_parse_xml_tags(self):
        from lllm.core.prompt import DefaultTagParser
        # Parser only requires "answer" and "reasoning"
        parser = DefaultTagParser(xml_tags=["answer", "reasoning"])
        content = "<answer>42</answer><reasoning>Because math.</reasoning>"
        result = parser.parse(content)
        self.assertEqual(result["xml_tags"]["answer"], ["42"])
        self.assertEqual(result["xml_tags"]["reasoning"], ["Because math."])

    def test_parse_multiple_same_tag(self):
        from lllm.core.prompt import DefaultTagParser
        # Only "answer" required
        parser = DefaultTagParser(xml_tags=["answer"])
        content = "<answer>first</answer> and <answer>second</answer>"
        result = parser.parse(content)
        self.assertEqual(result["xml_tags"]["answer"], ["first", "second"])

    def test_parse_md_tags(self):
        from lllm.core.prompt import DefaultTagParser
        # Only "python" md_tag required — no xml_tags
        parser = DefaultTagParser(md_tags=["python"])
        content = "```python\nprint('hello')\n```"
        result = parser.parse(content)
        self.assertIn("python", result["md_tags"])
        self.assertEqual(result["md_tags"]["python"][0], "print('hello')")

    def test_parse_signal_tags_true(self):
        from lllm.core.prompt import DefaultTagParser
        # Signal tags only — no required xml/md tags
        parser = DefaultTagParser(signal_tags=["DONE", "RETRY"])
        content = "I am finished. <DONE>"
        result = parser.parse(content)
        self.assertTrue(result["signal_tags"]["DONE"])
        self.assertFalse(result["signal_tags"]["RETRY"])

    def test_parse_raw_always_present(self):
        from lllm.core.prompt import DefaultTagParser
        # Empty parser — never raises
        parser = DefaultTagParser()
        content = "some raw content"
        result = parser.parse(content)
        self.assertEqual(result["raw"], content)

    def test_missing_xml_tag_raises_parse_error(self):
        from lllm.core.prompt import DefaultTagParser
        from lllm.core.const import ParseError
        # Parser requires both answer and reasoning
        parser = DefaultTagParser(xml_tags=["answer", "reasoning"])
        content = "<reasoning>thinking</reasoning>"  # missing <answer>
        with self.assertRaises(ParseError):
            parser.parse(content)

    def test_missing_md_tag_raises_parse_error(self):
        from lllm.core.prompt import DefaultTagParser
        from lllm.core.const import ParseError
        parser = DefaultTagParser(md_tags=["python"])
        content = "No code block here"
        with self.assertRaises(ParseError):
            parser.parse(content)

    def test_required_xml_tags_present_no_error(self):
        from lllm.core.prompt import DefaultTagParser
        parser = DefaultTagParser(
            xml_tags=["answer"],
            required_xml_tags=["answer"],
        )
        content = "<answer>yes</answer>"
        result = parser.parse(content)
        self.assertEqual(result["xml_tags"]["answer"], ["yes"])

    def test_required_xml_tags_missing_raises(self):
        from lllm.core.prompt import DefaultTagParser
        from lllm.core.const import ParseError
        parser = DefaultTagParser(
            xml_tags=["answer"],
            required_xml_tags=["answer"],
        )
        with self.assertRaises(ParseError):
            parser.parse("no answer tags")

    def test_empty_parser_no_tags(self):
        from lllm.core.prompt import DefaultTagParser
        parser = DefaultTagParser()
        result = parser.parse("raw content only")
        self.assertEqual(result["raw"], "raw content only")
        self.assertEqual(result["xml_tags"], {})
        self.assertEqual(result["md_tags"], {})
        self.assertEqual(result["signal_tags"], {})

    def test_multiline_xml_content(self):
        from lllm.core.prompt import DefaultTagParser
        parser = DefaultTagParser(xml_tags=["code"])
        content = "<code>\nline1\nline2\nline3\n</code>"
        result = parser.parse(content)
        self.assertIn("line1", result["xml_tags"]["code"][0])
        self.assertIn("line3", result["xml_tags"]["code"][0])

    def test_custom_subclass_can_extend(self):
        from lllm.core.prompt import DefaultTagParser
        from lllm.core.const import ParseError

        class StrictParser(DefaultTagParser):
            def parse(self, content, **kwargs):
                result = super().parse(content, **kwargs)
                if "error" in result.get("raw", "").lower():
                    raise ParseError("Content contains 'error'")
                return result

        parser = StrictParser(xml_tags=["answer"])
        with self.assertRaises(ParseError):
            parser.parse("<answer>error occurred</answer>")

        result = parser.parse("<answer>success</answer>")
        self.assertEqual(result["xml_tags"]["answer"], ["success"])


# ===========================================================================
# Function
# ===========================================================================


class TestFunction(unittest.TestCase):

    def _make_fn(self, **kwargs):
        from lllm.core.prompt import Function
        defaults = {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "units": {"type": "string", "description": "celsius or fahrenheit"},
            },
            "required": ["city"],
        }
        defaults.update(kwargs)
        return Function(**defaults)

    def test_basic_construction(self):
        fn = self._make_fn()
        self.assertEqual(fn.name, "get_weather")
        self.assertIsNone(fn.function)
        self.assertFalse(fn.linked)

    def test_link_function(self):
        fn = self._make_fn()
        fn.link_function(lambda city, units="celsius": f"Sunny in {city}")
        self.assertTrue(fn.linked)

    def test_link_non_callable_raises(self):
        fn = self._make_fn()
        with self.assertRaises(TypeError):
            fn.link_function("not a callable")

    def test_call_success(self):
        from lllm.core.const import FunctionCall
        fn = self._make_fn()
        fn.link_function(lambda city, units="celsius": f"22°C in {city}")
        fc = FunctionCall(id="c1", name="get_weather", arguments={"city": "Paris"})
        result = fn(fc)
        self.assertTrue(result.success)
        self.assertIn("Paris", result.result_str)

    def test_call_exception_captured(self):
        from lllm.core.const import FunctionCall
        fn = self._make_fn()
        fn.link_function(lambda city, units: 1 / 0)  # will raise ZeroDivisionError
        fc = FunctionCall(id="c1", name="get_weather", arguments={"city": "Paris", "units": "celsius"})
        result = fn(fc)
        self.assertIsNotNone(result.error_message)
        self.assertFalse(result.success)

    def test_call_unlinked_raises_assertion(self):
        from lllm.core.const import FunctionCall
        fn = self._make_fn()
        fc = FunctionCall(id="c1", name="get_weather", arguments={"city": "Paris"})
        with self.assertRaises(AssertionError):
            fn(fc)

    def test_to_tool_litellm(self):
        from lllm.core.const import Invokers
        fn = self._make_fn()
        tool = fn.to_tool(Invokers.LITELLM)
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "get_weather")
        self.assertIn("city", tool["function"]["parameters"]["properties"])
        self.assertEqual(tool["function"]["parameters"]["required"], ["city"])

    def test_to_tool_unsupported_invoker_raises(self):
        fn = self._make_fn()
        with self.assertRaises(NotImplementedError):
            fn.to_tool("unsupported_invoker")

    def test_from_callable_basic(self):
        from lllm.core.prompt import Function

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        fn = Function.from_callable(add)
        self.assertEqual(fn.name, "add")
        self.assertIn("x", fn.properties)
        self.assertIn("y", fn.properties)
        self.assertEqual(fn.properties["x"]["type"], "integer")
        self.assertIn("x", fn.required)
        self.assertIn("y", fn.required)

    def test_from_callable_with_defaults(self):
        from lllm.core.prompt import Function

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        fn = Function.from_callable(greet)
        self.assertIn("name", fn.required)
        self.assertNotIn("greeting", fn.required)
        self.assertIn("default:", fn.properties["greeting"].get("description", ""))

    def test_from_callable_optional_type(self):
        from lllm.core.prompt import Function

        def search(query: str, limit: Optional[int] = None) -> list:
            return []

        fn = Function.from_callable(search)
        self.assertEqual(fn.properties["limit"]["type"], "integer")
        self.assertNotIn("limit", fn.required)

    def test_from_callable_prop_desc(self):
        from lllm.core.prompt import Function

        def fn(city: str) -> str:
            return city

        result = Function.from_callable(fn, prop_desc={"city": "The city name"})
        self.assertIn("The city name", result.properties["city"].get("description", ""))

    def test_from_callable_custom_name_and_desc(self):
        from lllm.core.prompt import Function

        def fn(x: str) -> str:
            return x

        result = Function.from_callable(fn, name="custom_name", description="Custom description")
        self.assertEqual(result.name, "custom_name")
        self.assertEqual(result.description, "Custom description")

    def test_from_callable_no_type_hint_defaults_string(self):
        from lllm.core.prompt import Function

        def fn(x) -> str:
            return str(x)

        result = Function.from_callable(fn)
        self.assertEqual(result.properties["x"]["type"], "string")

    def test_custom_processor(self):
        from lllm.core.prompt import Function
        from lllm.core.const import FunctionCall

        def my_proc(result, fc):
            return f"[PROCESSED] {result}"

        def fn(x: str) -> str:
            return f"got {x}"

        func = Function.from_callable(fn, processor=my_proc)
        fc = FunctionCall(id="1", name="fn", arguments={"x": "hello"})
        result = func(fc)
        self.assertTrue(result.result_str.startswith("[PROCESSED]"))


# ===========================================================================
# @tool decorator
# ===========================================================================


class TestToolDecorator(unittest.TestCase):

    def test_basic_decorator(self):
        from lllm.core.prompt import tool, Function

        @tool(description="Get the current temperature")
        def get_temp(city: str, units: str = "celsius") -> str:
            return f"22 {units}"

        self.assertIsInstance(get_temp, Function)
        self.assertEqual(get_temp.name, "get_temp")
        self.assertEqual(get_temp.description, "Get the current temperature")
        self.assertTrue(get_temp.linked)

    def test_decorator_with_name_override(self):
        from lllm.core.prompt import tool, Function

        @tool(name="weather_api", description="Get weather")
        def _internal_fn(city: str) -> str:
            return city

        self.assertEqual(_internal_fn.name, "weather_api")

    def test_decorator_callable(self):
        from lllm.core.prompt import tool
        from lllm.core.const import FunctionCall

        @tool(description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            """Multiply a by b."""
            return a * b

        fc = FunctionCall(id="1", name="multiply", arguments={"a": 3, "b": 7})
        result = multiply(fc)
        self.assertTrue(result.success)
        self.assertEqual(result.result, 21)

    def test_decorator_uses_docstring_as_description(self):
        from lllm.core.prompt import tool, Function

        @tool()
        def compute(x: float) -> float:
            """Compute the square of x."""
            return x * x

        self.assertIn("square", compute.description)

    def test_decorator_prop_desc(self):
        from lllm.core.prompt import tool, Function

        @tool(prop_desc={"city": "The city to look up"})
        def lookup(city: str) -> dict:
            return {}

        self.assertIn("The city to look up", lookup.properties["city"]["description"])


# ===========================================================================
# MCP
# ===========================================================================


class TestMCP(unittest.TestCase):

    def test_basic_construction(self):
        from lllm.core.prompt import MCP
        mcp = MCP(server_label="tools", server_url="https://example.com/mcp")
        self.assertEqual(mcp.server_label, "tools")
        self.assertEqual(mcp.require_approval, "never")

    def test_valid_approval_values(self):
        from lllm.core.prompt import MCP
        for v in ("never", "manual", "auto"):
            MCP(server_label="x", server_url="https://x.com", require_approval=v)

    def test_invalid_approval_raises(self):
        from lllm.core.prompt import MCP
        with self.assertRaises(Exception):
            MCP(server_label="x", server_url="https://x.com", require_approval="always")

    def test_to_tool_litellm(self):
        from lllm.core.prompt import MCP
        from lllm.core.const import Invokers
        mcp = MCP(
            server_label="search",
            server_url="https://search.example.com",
            require_approval="never",
            allowed_tools=["web_search", "read"],
        )
        tool = mcp.to_tool(Invokers.LITELLM)
        self.assertEqual(tool["type"], "mcp")
        self.assertEqual(tool["server_label"], "search")
        self.assertIn("allowed_tools", tool)

    def test_to_tool_no_allowed_tools(self):
        from lllm.core.prompt import MCP
        from lllm.core.const import Invokers
        mcp = MCP(server_label="x", server_url="https://x.com")
        tool = mcp.to_tool(Invokers.LITELLM)
        self.assertNotIn("allowed_tools", tool)

    def test_to_tool_unsupported_returns_none(self):
        from lllm.core.prompt import MCP
        mcp = MCP(server_label="x", server_url="https://x.com")
        result = mcp.to_tool("unsupported")
        self.assertIsNone(result)


# ===========================================================================
# Prompt
# ===========================================================================


class TestPrompt(unittest.TestCase):

    def _make(self, **kwargs):
        from lllm.core.prompt import Prompt
        defaults = {"path": "test/prompt", "prompt": "Hello {name}!"}
        defaults.update(kwargs)
        return Prompt(**defaults)

    def test_basic_construction(self):
        p = self._make()
        self.assertEqual(p.path, "test/prompt")
        self.assertIsNotNone(p.handler)
        self.assertIsNotNone(p.renderer)

    def test_template_vars_extracted(self):
        p = self._make(prompt="{subject} and {verb} the {object}")
        self.assertEqual(p.template_vars, {"subject", "verb", "object"})

    def test_template_vars_no_vars(self):
        p = self._make(prompt="Static prompt with no variables.")
        self.assertEqual(p.template_vars, set())

    def test_template_vars_escaped_braces_not_included(self):
        p = self._make(prompt="Use {{escaped}} literal. Only {real} is a var.")
        self.assertEqual(p.template_vars, {"real"})

    def test_call_with_vars(self):
        p = self._make(prompt="Hello {name}!", prompt_kw=None)
        result = p(name="World")
        self.assertEqual(result, "Hello World!")

    def test_call_no_vars_no_kwargs(self):
        p = self._make(prompt="Static content")
        result = p()
        self.assertEqual(result, "Static content")

    def test_call_missing_vars_raises(self):
        p = self._make(prompt="Hello {name}, you are {age} years old.")
        with self.assertRaises(ValueError):
            p(name="Alice")  # missing 'age'

    def test_call_extra_kwargs_ok(self):
        """Extra kwargs are passed to renderer but template only uses declared vars."""
        p = self._make(prompt="Hello {name}!")
        result = p(name="Alice", extra="ignored")
        self.assertEqual(result, "Hello Alice!")

    def test_validate_args_returns_missing(self):
        p = self._make(prompt="{a} and {b}")
        missing = p.validate_args({"a": 1})
        self.assertEqual(missing, ["b"])

    def test_validate_args_all_present(self):
        p = self._make(prompt="{a} and {b}")
        missing = p.validate_args({"a": 1, "b": 2})
        self.assertEqual(missing, [])

    def test_parse_without_parser(self):
        p = self._make(prompt="Hello {name}!")
        result = p.parse("some output")
        self.assertEqual(result, {"raw": "some output"})

    def test_parse_with_parser(self):
        from lllm.core.prompt import Prompt, DefaultTagParser
        p = Prompt(
            path="test",
            prompt="Answer: {q}",
            parser=DefaultTagParser(xml_tags=["answer"]),
        )
        result = p.parse("<answer>42</answer>")
        self.assertEqual(result["xml_tags"]["answer"], ["42"])
        self.assertIn("raw", result)

    def test_link_function(self):
        from lllm.core.prompt import Prompt, Function
        fn = Function(name="get_data", description="Get data", properties={}, required=[])
        p = Prompt(path="test", prompt="Use get_data", function_list=[fn])
        p.link_function("get_data", lambda: {"data": 42})
        self.assertTrue(p.functions["get_data"].linked)

    def test_link_function_not_found_raises(self):
        p = self._make()
        with self.assertRaises(KeyError):
            p.link_function("nonexistent", lambda: None)

    def test_get_function(self):
        from lllm.core.prompt import Prompt, Function
        fn = Function(name="my_fn", description="desc", properties={}, required=[])
        p = Prompt(path="test", prompt="test", function_list=[fn])
        self.assertIs(p.get_function("my_fn"), fn)

    def test_get_function_not_found_raises(self):
        p = self._make()
        with self.assertRaises(KeyError):
            p.get_function("does_not_exist")

    def test_functions_dict_keyed_by_name(self):
        from lllm.core.prompt import Prompt, Function
        fn1 = Function(name="f1", description="d1", properties={}, required=[])
        fn2 = Function(name="f2", description="d2", properties={}, required=[])
        p = Prompt(path="test", prompt="test", function_list=[fn1, fn2])
        self.assertIn("f1", p.functions)
        self.assertIn("f2", p.functions)

    def test_allow_web_search_false_by_default(self):
        p = self._make()
        self.assertFalse(p.allow_web_search)

    def test_allow_web_search_true_via_addon_args(self):
        p = self._make(addon_args={"web_search": True})
        self.assertTrue(p.allow_web_search)

    def test_computer_use_config_empty_by_default(self):
        p = self._make()
        self.assertEqual(p.computer_use_config, {})

    def test_computer_use_config_from_addon_args(self):
        p = self._make(addon_args={"computer_use": {"display": "1024x768"}})
        self.assertEqual(p.computer_use_config, {"display": "1024x768"})

    def test_extend_requires_path(self):
        p = self._make()
        with self.assertRaises(ValueError):
            p.extend(prompt="new prompt")

    def test_extend_creates_child(self):
        p = self._make(prompt="Parent: {topic}")
        child = p.extend(path="child/prompt", prompt="Child: {topic}")
        self.assertEqual(child.path, "child/prompt")
        self.assertEqual(child.prompt, "Child: {topic}")
        # Inherits other fields
        self.assertIs(child.handler, p.handler)

    def test_extend_overrides_function_list(self):
        from lllm.core.prompt import Prompt, Function
        fn = Function(name="f1", description="d", properties={}, required=[])
        p = Prompt(path="parent", prompt="test", function_list=[fn])
        child = p.extend(path="child", function_list=[])
        self.assertEqual(child.function_list, [])

    def test_info_dict_structure(self):
        p = self._make(prompt="Hello {name}!")
        info = p.info_dict()
        self.assertIn("path", info)
        self.assertIn("prompt_hash", info)
        self.assertIn("metadata", info)
        self.assertIn("functions", info)
        self.assertIn("has_parser", info)
        self.assertEqual(info["has_parser"], False)

    def test_info_dict_hash_stable(self):
        p = self._make(prompt="Same content")
        info1 = p.info_dict()
        info2 = p.info_dict()
        self.assertEqual(info1["prompt_hash"], info2["prompt_hash"])

    def test_register_mcp_server(self):
        from lllm.core.prompt import MCP
        p = self._make()
        mcp = MCP(server_label="tools", server_url="https://tools.example.com")
        p.register_mcp_server(mcp)
        self.assertIn("tools", p.mcp_servers)

    def test_mcp_servers_dict_from_list(self):
        from lllm.core.prompt import Prompt, MCP
        m1 = MCP(server_label="s1", server_url="https://s1.example.com")
        m2 = MCP(server_label="s2", server_url="https://s2.example.com")
        p = Prompt(path="test", prompt="test", mcp_servers_list=[m1, m2])
        self.assertIn("s1", p.mcp_servers)
        self.assertIn("s2", p.mcp_servers)

    def test_on_exception_returns_prompt(self):
        p = self._make()
        mock_session = MagicMock()
        result = p.on_exception(mock_session)
        self.assertIsNotNone(result)

    def test_on_interrupt_returns_prompt(self):
        p = self._make()
        mock_session = MagicMock()
        result = p.on_interrupt(mock_session)
        self.assertIsNotNone(result)

    def test_on_interrupt_final_returns_prompt(self):
        p = self._make()
        mock_session = MagicMock()
        result = p.on_interrupt_final(mock_session)
        self.assertIsNotNone(result)


# ===========================================================================
# DefaultSimpleHandler
# ===========================================================================


class TestDefaultSimpleHandler(unittest.TestCase):

    def setUp(self):
        from lllm.core.prompt import DefaultSimpleHandler, Prompt
        self.handler = DefaultSimpleHandler()
        self.prompt = Prompt(path="base", prompt="Base prompt: {task}")

    def test_on_exception_returns_extended_prompt(self):
        mock_session = MagicMock()
        result = self.handler.on_exception(self.prompt, mock_session)
        self.assertIn("__base_exception", result.path)

    def test_on_interrupt_returns_extended_prompt(self):
        mock_session = MagicMock()
        result = self.handler.on_interrupt(self.prompt, mock_session)
        self.assertIn("__base_interrupt", result.path)

    def test_on_interrupt_final_returns_extended_prompt(self):
        mock_session = MagicMock()
        result = self.handler.on_interrupt_final(self.prompt, mock_session)
        self.assertIn("__base_interrupt_final", result.path)

    def test_exception_prompt_inherits_tools(self):
        from lllm.core.prompt import Prompt, Function
        fn = Function(name="tool", description="t", properties={}, required=[])
        p = Prompt(path="parent", prompt="test", function_list=[fn])
        handler = self.handler
        result = handler.on_exception(p, MagicMock())
        self.assertEqual(result.function_list, [fn])

    def test_interrupt_final_has_no_tools(self):
        from lllm.core.prompt import Prompt, Function
        fn = Function(name="tool", description="t", properties={}, required=[])
        p = Prompt(path="parent", prompt="test", function_list=[fn])
        result = self.handler.on_interrupt_final(p, MagicMock())
        self.assertEqual(result.function_list, [])

    def test_custom_exception_msg(self):
        from lllm.core.prompt import DefaultSimpleHandler, Prompt
        handler = DefaultSimpleHandler(exception_msg="Custom error: {error_message}")
        p = Prompt(path="base", prompt="test")
        result = handler.on_exception(p, MagicMock())
        self.assertIn("Custom error:", result.prompt)

    def test_handler_with_prompt_object_returns_it_directly(self):
        from lllm.core.prompt import DefaultSimpleHandler, Prompt
        # _resolve_handler supports Union[str, Prompt]; bypass Pydantic typing
        # by calling _resolve_handler directly with a Prompt instance
        handler = DefaultSimpleHandler()
        p = Prompt(path="base", prompt="test")
        exception_prompt = Prompt(path="my/exception", prompt="fixed exception handler")
        # Call _resolve_handler directly with Prompt handler
        result = handler._resolve_handler(p, exception_prompt, suffix="exception", inherit_tools=True)
        self.assertIs(result, exception_prompt)


# ===========================================================================
# AgentCallSession
# ===========================================================================


class TestAgentCallSession(unittest.TestCase):

    def _make_session(self):
        from lllm.core.prompt import AgentCallSession
        return AgentCallSession(
            agent_name="test_agent",
            max_exception_retry=3,
            max_interrupt_steps=5,
            max_llm_recall=2,
        )

    def _make_invoke_result(self, tokens=0):
        from lllm.core.const import InvokeResult, InvokeCost
        from lllm.core.dialog import Message
        from lllm.core.const import Roles
        msg = Message(role=Roles.ASSISTANT, content="reply", name="agent",
                      usage={"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens})
        return InvokeResult(message=msg)

    def test_initial_state(self):
        s = self._make_session()
        self.assertEqual(s.state, "initial")
        self.assertIsNone(s.delivery)
        self.assertEqual(s.exception_retries_count, 0)
        self.assertEqual(s.llm_recalls_count, 0)

    def test_new_invoke_trace(self):
        s = self._make_session()
        ir = self._make_invoke_result(100)
        s.new_invoke_trace(ir, interrupt_step=0)
        self.assertIn(0, s.invoke_traces)
        self.assertEqual(len(s.invoke_traces[0]), 1)

    def test_exception_increments_count(self):
        s = self._make_session()
        s.exception(ValueError("test error"), interrupt_step=0)
        s.exception(ValueError("another"), interrupt_step=0)
        self.assertEqual(s.exception_retries_count, 2)
        self.assertEqual(s.state, "exception")

    def test_exception_different_steps(self):
        s = self._make_session()
        s.exception(ValueError("e1"), interrupt_step=0)
        s.exception(ValueError("e2"), interrupt_step=1)
        self.assertEqual(s.exception_retries_count, 2)

    def test_interrupt_records(self):
        from lllm.core.const import FunctionCall
        s = self._make_session()
        fc = FunctionCall(id="1", name="fn", arguments={})
        s.interrupt([fc], interrupt_step=0)
        self.assertIn(0, s.interrupts)
        self.assertEqual(s.state, "interrupt")

    def test_llm_recall_increments(self):
        s = self._make_session()
        s.llm_recall(ValueError("recall"), interrupt_step=0)
        self.assertEqual(s.llm_recalls_count, 1)
        self.assertEqual(s.state, "llm_recall")

    def test_success_sets_delivery(self):
        from lllm.core.const import Roles
        from lllm.core.dialog import Message
        s = self._make_session()
        msg = Message(role=Roles.ASSISTANT, content="final", name="agent")
        s.success(msg)
        self.assertEqual(s.state, "success")
        self.assertIs(s.delivery, msg)

    def test_failure_sets_state(self):
        s = self._make_session()
        s.failure()
        self.assertEqual(s.state, "failure")

    def test_reach_max_exception_retry(self):
        s = self._make_session()
        for i in range(3):
            s.exception(ValueError(f"e{i}"), interrupt_step=0)
        self.assertTrue(s.reach_max_exception_retry)

    def test_not_reach_max_exception_retry(self):
        s = self._make_session()
        s.exception(ValueError("one"), interrupt_step=0)
        self.assertFalse(s.reach_max_exception_retry)

    def test_reach_max_llm_recall(self):
        s = self._make_session()
        s.llm_recall(ValueError("r1"), interrupt_step=0)
        s.llm_recall(ValueError("r2"), interrupt_step=1)
        self.assertTrue(s.reach_max_llm_recall)

    def test_reach_max_interrupt_steps(self):
        from lllm.core.const import FunctionCall
        s = self._make_session()
        for i in range(5):
            s.interrupt([FunctionCall(id=str(i), name="fn", arguments={})], interrupt_step=i)
        self.assertTrue(s.reach_max_interrupt_steps)

    def test_cost_sums_traces(self):
        s = self._make_session()
        ir1 = self._make_invoke_result(100)
        ir2 = self._make_invoke_result(200)
        s.new_invoke_trace(ir1, interrupt_step=0)
        s.new_invoke_trace(ir2, interrupt_step=1)
        self.assertEqual(s.cost.prompt_tokens, 300)

    def test_cost_empty_traces(self):
        from lllm.core.const import InvokeCost
        s = self._make_session()
        cost = s.cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.total_tokens, 0)


# ===========================================================================
# module-level register_prompt
# ===========================================================================


class TestRegisterPrompt(unittest.TestCase):

    def test_register_to_default_runtime(self):
        from lllm.core.prompt import Prompt, register_prompt
        from lllm.core.runtime import get_default_runtime
        p = Prompt(path="__test_register__", prompt="test")
        register_prompt(p, overwrite=True)
        rt = get_default_runtime()
        self.assertTrue(rt.has("__test_register__"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
