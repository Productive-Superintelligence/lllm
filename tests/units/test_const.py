"""
Tests for lllm/core/const.py
Covers: Roles, Invokers, Modalities, APITypes, FunctionCall, ParseError,
        InvokeResult, InvokeCost, LLM_SIDE_ROLES.
"""
import unittest
from unittest.mock import MagicMock


class TestRolesEnum(unittest.TestCase):

    def test_values(self):
        from lllm.core.const import Roles
        self.assertEqual(Roles.SYSTEM.value, "system")
        self.assertEqual(Roles.ASSISTANT.value, "assistant")
        self.assertEqual(Roles.USER.value, "user")
        self.assertEqual(Roles.TOOL.value, "tool")
        self.assertEqual(Roles.TOOL_CALL.value, "tool_call")

    def test_msg_value_system_maps_to_developer(self):
        from lllm.core.const import Roles
        self.assertEqual(Roles.SYSTEM.msg_value, "developer")

    def test_msg_value_others_unchanged(self):
        from lllm.core.const import Roles
        self.assertEqual(Roles.ASSISTANT.msg_value, "assistant")
        self.assertEqual(Roles.USER.msg_value, "user")
        self.assertEqual(Roles.TOOL.msg_value, "tool")
        self.assertEqual(Roles.TOOL_CALL.msg_value, "tool_call")

    def test_is_str_enum(self):
        from lllm.core.const import Roles
        self.assertIsInstance(Roles.USER, str)
        self.assertEqual(Roles.USER, "user")

    def test_membership(self):
        from lllm.core.const import Roles
        self.assertIn(Roles.SYSTEM, list(Roles))
        self.assertEqual(len(list(Roles)), 5)


class TestInvokersEnum(unittest.TestCase):

    def test_litellm_value(self):
        from lllm.core.const import Invokers
        self.assertEqual(Invokers.LITELLM.value, "litellm")

    def test_is_str_enum(self):
        from lllm.core.const import Invokers
        self.assertIsInstance(Invokers.LITELLM, str)


class TestModalitiesEnum(unittest.TestCase):

    def test_all_values(self):
        from lllm.core.const import Modalities
        self.assertEqual(Modalities.TEXT.value, "text")
        self.assertEqual(Modalities.IMAGE.value, "image")
        self.assertEqual(Modalities.AUDIO.value, "audio")
        self.assertEqual(Modalities.FUNCTION_CALL.value, "function_call")

    def test_count(self):
        from lllm.core.const import Modalities
        self.assertEqual(len(list(Modalities)), 4)


class TestAPITypesEnum(unittest.TestCase):

    def test_values(self):
        from lllm.core.const import APITypes
        self.assertEqual(APITypes.COMPLETION.value, "completion")
        self.assertEqual(APITypes.RESPONSE.value, "response")


class TestLLMSideRoles(unittest.TestCase):

    def test_contains_assistant_and_tool_call(self):
        from lllm.core.const import LLM_SIDE_ROLES, Roles
        self.assertIn(Roles.ASSISTANT, LLM_SIDE_ROLES)
        self.assertIn(Roles.TOOL_CALL, LLM_SIDE_ROLES)
        self.assertNotIn(Roles.USER, LLM_SIDE_ROLES)
        self.assertNotIn(Roles.SYSTEM, LLM_SIDE_ROLES)


class TestParseError(unittest.TestCase):

    def test_basic_construction(self):
        from lllm.core.const import ParseError
        e = ParseError("something went wrong")
        self.assertEqual(e.message, "something went wrong")
        self.assertEqual(e.detail, "")
        self.assertIsInstance(e, Exception)

    def test_with_detail(self):
        from lllm.core.const import ParseError
        e = ParseError("parse failed", detail="line 42: unexpected token")
        self.assertEqual(e.message, "parse failed")
        self.assertEqual(e.detail, "line 42: unexpected token")

    def test_str_representation(self):
        from lllm.core.const import ParseError
        e = ParseError("parse failed")
        self.assertEqual(str(e), "parse failed")

    def test_is_exception(self):
        from lllm.core.const import ParseError
        with self.assertRaises(ParseError):
            raise ParseError("test")


class TestFunctionCall(unittest.TestCase):

    def _make(self, **kwargs):
        from lllm.core.const import FunctionCall
        defaults = {"id": "call_abc", "name": "get_weather", "arguments": {"city": "Paris"}}
        defaults.update(kwargs)
        return FunctionCall(**defaults)

    def test_basic_construction(self):
        fc = self._make()
        self.assertEqual(fc.id, "call_abc")
        self.assertEqual(fc.name, "get_weather")
        self.assertEqual(fc.arguments, {"city": "Paris"})
        self.assertIsNone(fc.result)
        self.assertIsNone(fc.result_str)
        self.assertIsNone(fc.error_message)

    def test_success_true_when_result_str_set_no_error(self):
        fc = self._make(result_str="Sunny, 22°C")
        self.assertTrue(fc.success)

    def test_success_false_when_no_result_str(self):
        fc = self._make()
        self.assertFalse(fc.success)

    def test_success_false_when_error_message_set(self):
        fc = self._make(result_str="data", error_message="timeout")
        self.assertFalse(fc.success)

    def test_str_with_success(self):
        fc = self._make(result_str="Sunny")
        s = str(fc)
        self.assertIn("get_weather", s)
        self.assertIn("Paris", s)
        self.assertIn("Sunny", s)

    def test_str_without_success(self):
        fc = self._make()
        s = str(fc)
        self.assertIn("get_weather", s)
        self.assertNotIn("Return:", s)

    def test_equals_same_args(self):
        from lllm.core.const import FunctionCall
        a = FunctionCall(id="1", name="fn", arguments={"x": 1, "y": 2})
        b = FunctionCall(id="2", name="fn", arguments={"x": 1, "y": 2})
        self.assertTrue(a.equals(b))

    def test_equals_different_name(self):
        from lllm.core.const import FunctionCall
        a = FunctionCall(id="1", name="fn_a", arguments={"x": 1})
        b = FunctionCall(id="2", name="fn_b", arguments={"x": 1})
        self.assertFalse(a.equals(b))

    def test_equals_different_keys(self):
        from lllm.core.const import FunctionCall
        a = FunctionCall(id="1", name="fn", arguments={"x": 1})
        b = FunctionCall(id="2", name="fn", arguments={"y": 1})
        self.assertFalse(a.equals(b))

    def test_equals_different_values(self):
        from lllm.core.const import FunctionCall
        a = FunctionCall(id="1", name="fn", arguments={"x": 1})
        b = FunctionCall(id="2", name="fn", arguments={"x": 99})
        self.assertFalse(a.equals(b))

    def test_equals_empty_arguments(self):
        from lllm.core.const import FunctionCall
        a = FunctionCall(id="1", name="fn", arguments={})
        b = FunctionCall(id="2", name="fn", arguments={})
        self.assertTrue(a.equals(b))

    def test_is_repeated_found(self):
        from lllm.core.const import FunctionCall
        fc = FunctionCall(id="1", name="fn", arguments={"x": 1})
        history = [
            FunctionCall(id="0", name="fn", arguments={"x": 0}),
            FunctionCall(id="2", name="fn", arguments={"x": 1}),
        ]
        self.assertTrue(fc.is_repeated(history))

    def test_is_repeated_not_found(self):
        from lllm.core.const import FunctionCall
        fc = FunctionCall(id="1", name="fn", arguments={"x": 99})
        history = [FunctionCall(id="2", name="fn", arguments={"x": 1})]
        self.assertFalse(fc.is_repeated(history))

    def test_is_repeated_empty_history(self):
        from lllm.core.const import FunctionCall
        fc = FunctionCall(id="1", name="fn", arguments={"x": 1})
        self.assertFalse(fc.is_repeated([]))


class TestInvokeCost(unittest.TestCase):

    def test_defaults_zero(self):
        from lllm.core.const import InvokeCost
        c = InvokeCost()
        self.assertEqual(c.prompt_tokens, 0)
        self.assertEqual(c.completion_tokens, 0)
        self.assertEqual(c.total_tokens, 0)
        self.assertEqual(c.cost, 0.0)

    def test_str_contains_token_counts(self):
        from lllm.core.const import InvokeCost
        c = InvokeCost(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        s = str(c)
        self.assertIn("150", s)
        self.assertIn("100", s)
        self.assertIn("50", s)

    def test_str_shows_rates_when_nonzero(self):
        from lllm.core.const import InvokeCost
        c = InvokeCost(input_cost_per_token=0.0000015, output_cost_per_token=0.000006, cost=0.00025)
        s = str(c)
        self.assertIn("In:", s)
        self.assertIn("Out:", s)

    def test_str_no_rates_when_zero(self):
        from lllm.core.const import InvokeCost
        c = InvokeCost(prompt_tokens=10)
        s = str(c)
        self.assertNotIn("Rates:", s)

    def test_add_tokens_sum(self):
        from lllm.core.const import InvokeCost
        a = InvokeCost(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.01)
        b = InvokeCost(prompt_tokens=200, completion_tokens=80, total_tokens=280, cost=0.02)
        result = a + b
        self.assertEqual(result.prompt_tokens, 300)
        self.assertEqual(result.completion_tokens, 130)
        self.assertEqual(result.total_tokens, 430)
        self.assertAlmostEqual(result.cost, 0.03)

    def test_add_zeroes_rates(self):
        """Rates are NOT additive — they should be zeroed on aggregation."""
        from lllm.core.const import InvokeCost
        a = InvokeCost(input_cost_per_token=0.001)
        b = InvokeCost(input_cost_per_token=0.002)
        result = a + b
        self.assertEqual(result.input_cost_per_token, 0.0)
        self.assertEqual(result.output_cost_per_token, 0.0)
        self.assertEqual(result.cache_read_input_token_cost, 0.0)

    def test_add_all_fields(self):
        from lllm.core.const import InvokeCost
        a = InvokeCost(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            cached_prompt_tokens=10, reasoning_tokens=5,
            audio_prompt_tokens=3, audio_completion_tokens=2,
            prompt_cost=0.001, completion_cost=0.002, cost=0.003,
        )
        b = InvokeCost(
            prompt_tokens=200, completion_tokens=100, total_tokens=300,
            cached_prompt_tokens=20, reasoning_tokens=10,
            audio_prompt_tokens=6, audio_completion_tokens=4,
            prompt_cost=0.002, completion_cost=0.004, cost=0.006,
        )
        r = a + b
        self.assertEqual(r.prompt_tokens, 300)
        self.assertEqual(r.cached_prompt_tokens, 30)
        self.assertEqual(r.reasoning_tokens, 15)
        self.assertEqual(r.audio_prompt_tokens, 9)
        self.assertEqual(r.audio_completion_tokens, 6)
        self.assertAlmostEqual(r.prompt_cost, 0.003)
        self.assertAlmostEqual(r.cost, 0.009)


class TestInvokeResult(unittest.TestCase):

    def test_no_errors(self):
        from lllm.core.const import InvokeResult
        ir = InvokeResult()
        self.assertFalse(ir.has_errors)
        self.assertEqual(ir.error_message, "")

    def test_with_errors(self):
        from lllm.core.const import InvokeResult
        e1 = ValueError("bad value")
        e2 = RuntimeError("crash")
        ir = InvokeResult(execution_errors=[e1, e2])
        self.assertTrue(ir.has_errors)
        msg = ir.error_message
        self.assertIn("bad value", msg)
        self.assertIn("crash", msg)

    def test_cost_no_message(self):
        from lllm.core.const import InvokeResult, InvokeCost
        ir = InvokeResult()
        cost = ir.cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.total_tokens, 0)

    def test_cost_with_message(self):
        from lllm.core.const import InvokeResult
        from lllm.core.dialog import Message
        from lllm.core.const import Roles
        msg = Message(
            role=Roles.ASSISTANT, content="hello", name="agent",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        ir = InvokeResult(message=msg)
        self.assertEqual(ir.cost.total_tokens, 150)

    def test_model_args_default_empty(self):
        from lllm.core.const import InvokeResult
        ir = InvokeResult()
        self.assertEqual(ir.model_args, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
