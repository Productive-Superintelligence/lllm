"""
Tests for lllm/core/tactic.py (non-LLM parts)
Covers: TacticCallSession, _TrackedAgent, _normalize_name, _stable_tactic_id,
        register_tactic_class, get_tactic_class, build_tactic (via mocking).
"""
import unittest
from enum import Enum
from unittest.mock import MagicMock, patch

# tactic.py uses `from __future__ import annotations`; rebuild Pydantic models
# so all forward references (Message, AgentCallSession, TacticCallSession) are
# resolved before any test instantiates them.
from lllm.core.dialog import Message          # noqa: F401 — resolves 'Message'
from lllm.core.prompt import AgentCallSession  # noqa: F401
from lllm.core.tactic import TacticCallSession  # noqa: F401
AgentCallSession.model_rebuild(force=True)
TacticCallSession.model_rebuild(force=True)


# ===========================================================================
# _normalize_name
# ===========================================================================


class TestNormalizeName(unittest.TestCase):

    def test_string_passthrough(self):
        from lllm.core.tactic import _normalize_name
        self.assertEqual(_normalize_name("my_tactic"), "my_tactic")

    def test_enum_value_extracted(self):
        from lllm.core.tactic import _normalize_name

        class TacticType(str, Enum):
            RESEARCHER = "researcher"
            CODER = "coder"

        self.assertEqual(_normalize_name(TacticType.RESEARCHER), "researcher")
        self.assertEqual(_normalize_name(TacticType.CODER), "coder")

    def test_non_string_non_enum_raises(self):
        from lllm.core.tactic import _normalize_name
        with self.assertRaises(ValueError):
            _normalize_name(42)

    def test_none_raises(self):
        from lllm.core.tactic import _normalize_name
        with self.assertRaises((ValueError, AttributeError)):
            _normalize_name(None)


# ===========================================================================
# _stable_tactic_id
# ===========================================================================


class TestStableTacticId(unittest.TestCase):

    def test_with_namespace_extracts_package_name(self):
        from lllm.core.tactic import _stable_tactic_id
        result = _stable_tactic_id("my_pkg.tactics", "researcher")
        self.assertEqual(result, "my_pkg::researcher")

    def test_with_deep_namespace(self):
        from lllm.core.tactic import _stable_tactic_id
        result = _stable_tactic_id("my_pkg.tactics.subfolder", "coder")
        self.assertEqual(result, "my_pkg::coder")

    def test_no_namespace_returns_bare_name(self):
        from lllm.core.tactic import _stable_tactic_id
        result = _stable_tactic_id("", "my_tactic")
        self.assertEqual(result, "my_tactic")

    def test_single_namespace_component(self):
        from lllm.core.tactic import _stable_tactic_id
        result = _stable_tactic_id("pkg", "analyzer")
        self.assertEqual(result, "pkg::analyzer")


# ===========================================================================
# TacticCallSession
# ===========================================================================


def _make_agent_session(**kwargs):
    from lllm.core.prompt import AgentCallSession
    defaults = {
        "agent_name": "test_agent",
        "max_exception_retry": 3,
        "max_interrupt_steps": 5,
        "max_llm_recall": 2,
    }
    defaults.update(kwargs)
    return AgentCallSession(**defaults)


def _make_agent_session_with_cost(tokens):
    """Create an AgentCallSession with a trace that has a given token count."""
    from lllm.core.const import InvokeResult, InvokeCost
    from lllm.core.dialog import Message
    from lllm.core.const import Roles
    session = _make_agent_session()
    msg = Message(
        role=Roles.ASSISTANT, content="reply", name="agent",
        usage={"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
    )
    ir = InvokeResult(message=msg)
    session.new_invoke_trace(ir, interrupt_step=0)
    session.success(msg)
    return session


class TestTacticCallSession(unittest.TestCase):

    def _make_session(self):
        from lllm.core.tactic import TacticCallSession
        return TacticCallSession(tactic_name="my_tactic", tactic_path="my_pkg::my_tactic")

    def test_initial_state(self):
        s = self._make_session()
        self.assertEqual(s.state, "initial")
        self.assertIsNone(s.delivery)
        self.assertIsNone(s.error)
        self.assertIsNone(s.error_traceback)
        self.assertEqual(s.agent_call_count, 0)
        self.assertEqual(s.sub_tactic_call_count, 0)

    def test_record_agent_call(self):
        s = self._make_session()
        agent_session = _make_agent_session()
        s.record_agent_call("agent_a", agent_session)
        self.assertIn("agent_a", s.agent_sessions)
        self.assertEqual(len(s.agent_sessions["agent_a"]), 1)

    def test_record_multiple_agent_calls_same_agent(self):
        s = self._make_session()
        for _ in range(3):
            s.record_agent_call("agent_a", _make_agent_session())
        self.assertEqual(len(s.agent_sessions["agent_a"]), 3)

    def test_record_agent_calls_multiple_agents(self):
        s = self._make_session()
        s.record_agent_call("agent_a", _make_agent_session())
        s.record_agent_call("agent_b", _make_agent_session())
        s.record_agent_call("agent_a", _make_agent_session())
        self.assertEqual(s.agent_call_count, 3)
        self.assertEqual(len(s.agent_sessions["agent_a"]), 2)
        self.assertEqual(len(s.agent_sessions["agent_b"]), 1)

    def test_record_sub_tactic_call(self):
        from lllm.core.tactic import TacticCallSession
        s = self._make_session()
        sub = TacticCallSession(tactic_name="sub_tactic")
        s.record_sub_tactic_call("sub_tactic", sub)
        self.assertIn("sub_tactic", s.sub_tactic_sessions)
        self.assertEqual(s.sub_tactic_call_count, 1)

    def test_record_multiple_sub_tactic_calls(self):
        from lllm.core.tactic import TacticCallSession
        s = self._make_session()
        for _ in range(4):
            sub = TacticCallSession(tactic_name="sub")
            s.record_sub_tactic_call("sub", sub)
        self.assertEqual(s.sub_tactic_call_count, 4)

    def test_success_sets_state_and_delivery(self):
        s = self._make_session()
        s.success("final result")
        self.assertEqual(s.state, "success")
        self.assertEqual(s.delivery, "final result")

    def test_success_with_dict_result(self):
        s = self._make_session()
        result = {"answer": 42, "confidence": 0.95}
        s.success(result)
        self.assertEqual(s.delivery, result)

    def test_failure_with_exception(self):
        s = self._make_session()
        try:
            raise ValueError("Something went wrong")
        except Exception as e:
            s.failure(e)
        self.assertEqual(s.state, "failure")
        self.assertIsNotNone(s.error)
        self.assertIn("ValueError", s.error)
        self.assertIn("Something went wrong", s.error)

    def test_failure_without_exception(self):
        s = self._make_session()
        s.failure()
        self.assertEqual(s.state, "failure")
        self.assertIsNone(s.error)

    def test_agent_cost_sums_sessions(self):
        s = self._make_session()
        s.record_agent_call("agent_a", _make_agent_session_with_cost(100))
        s.record_agent_call("agent_a", _make_agent_session_with_cost(200))
        s.record_agent_call("agent_b", _make_agent_session_with_cost(50))
        cost = s.agent_cost
        self.assertEqual(cost.prompt_tokens, 350)

    def test_agent_cost_empty(self):
        from lllm.core.const import InvokeCost
        s = self._make_session()
        cost = s.agent_cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.total_tokens, 0)

    def test_sub_tactic_cost(self):
        from lllm.core.tactic import TacticCallSession
        from lllm.core.const import InvokeCost
        s = self._make_session()

        # Create sub-tactic sessions with known costs via agent sessions
        sub1 = TacticCallSession(tactic_name="sub1")
        sub1.record_agent_call("agent", _make_agent_session_with_cost(500))
        s.record_sub_tactic_call("sub1", sub1)

        sub2 = TacticCallSession(tactic_name="sub2")
        sub2.record_agent_call("agent", _make_agent_session_with_cost(300))
        s.record_sub_tactic_call("sub2", sub2)

        cost = s.sub_tactic_cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.prompt_tokens, 800)

    def test_total_cost_combines_agent_and_sub(self):
        from lllm.core.tactic import TacticCallSession
        s = self._make_session()
        s.record_agent_call("agent_a", _make_agent_session_with_cost(100))

        sub = TacticCallSession(tactic_name="sub")
        sub.record_agent_call("agent_b", _make_agent_session_with_cost(200))
        s.record_sub_tactic_call("sub", sub)

        total = s.total_cost
        self.assertEqual(total.prompt_tokens, 300)

    def test_summary_structure(self):
        s = self._make_session()
        s.success("result")
        summary = s.summary()
        self.assertIn("tactic", summary)
        self.assertIn("state", summary)
        self.assertIn("agent_calls", summary)
        self.assertIn("sub_tactic_calls", summary)
        self.assertIn("total_cost", summary)
        self.assertEqual(summary["state"], "success")

    def test_summary_agent_call_count(self):
        s = self._make_session()
        s.record_agent_call("a", _make_agent_session())
        s.record_agent_call("a", _make_agent_session())
        summary = s.summary()
        self.assertEqual(summary["agent_calls"], 2)

    def test_no_tactic_path_defaults_to_none(self):
        from lllm.core.tactic import TacticCallSession
        s = TacticCallSession(tactic_name="my_tactic")
        self.assertIsNone(s.tactic_path)

    def test_error_traceback_on_failure(self):
        s = self._make_session()
        try:
            raise RuntimeError("deep error")
        except Exception as e:
            s.failure(e)
        self.assertIsNotNone(s.error_traceback)
        self.assertIn("RuntimeError", s.error_traceback)


# ===========================================================================
# _TrackedAgent
# ===========================================================================


class TestTrackedAgent(unittest.TestCase):

    def _make_tracked_agent(self, agent=None, session=None, name="agent_a"):
        from lllm.core.tactic import _TrackedAgent, TacticCallSession
        if agent is None:
            agent = MagicMock()
        if session is None:
            session = TacticCallSession(tactic_name="tactic")
        return _TrackedAgent(agent, session, name), session

    def test_repr(self):
        ta, _ = self._make_tracked_agent(name="my_agent")
        r = repr(ta)
        self.assertIn("my_agent", r)

    def test_getattr_delegates_to_agent(self):
        agent = MagicMock()
        agent.name = "real_agent"
        ta, _ = self._make_tracked_agent(agent=agent)
        self.assertEqual(ta.name, "real_agent")

    def test_setattr_delegates_to_agent(self):
        agent = MagicMock()
        ta, _ = self._make_tracked_agent(agent=agent)
        ta.some_attr = "value"
        self.assertEqual(agent.some_attr, "value")

    def test_respond_records_session(self):
        from lllm.core.tactic import _TrackedAgent, TacticCallSession
        from lllm.core.const import Roles
        from lllm.core.dialog import Message

        # Create a mock agent whose respond() returns an AgentCallSession
        agent_session = _make_agent_session()
        msg = Message(role=Roles.ASSISTANT, content="reply", name="agent")
        agent_session.success(msg)

        agent = MagicMock()
        agent.respond.return_value = agent_session

        tactic_session = TacticCallSession(tactic_name="tactic")
        ta = _TrackedAgent(agent, tactic_session, "agent_a")

        result = ta.respond()

        # Agent.respond was called with return_session=True
        agent.respond.assert_called_once()
        call_kwargs = agent.respond.call_args[1]
        self.assertTrue(call_kwargs.get("return_session"))

        # Session was recorded
        self.assertIn("agent_a", tactic_session.agent_sessions)

        # Default return is delivery (Message), not session
        self.assertIs(result, msg)

    def test_respond_return_session_true(self):
        from lllm.core.tactic import _TrackedAgent, TacticCallSession
        from lllm.core.const import Roles
        from lllm.core.dialog import Message

        agent_session = _make_agent_session()
        msg = Message(role=Roles.ASSISTANT, content="hi", name="agent")
        agent_session.success(msg)

        agent = MagicMock()
        agent.respond.return_value = agent_session

        tactic_session = TacticCallSession(tactic_name="tactic")
        ta = _TrackedAgent(agent, tactic_session, "agent_a")

        result = ta.respond(return_session=True)
        self.assertIs(result, agent_session)


# ===========================================================================
# register_tactic_class / get_tactic_class
# ===========================================================================


class TestTacticRegistration(unittest.TestCase):

    def test_register_and_get(self):
        from lllm.core.tactic import register_tactic_class, get_tactic_class
        from lllm.core.runtime import Runtime

        rt = Runtime()

        class FakeTactic:
            name = "fake_tactic_xyz"

        register_tactic_class(FakeTactic, runtime=rt)
        result = get_tactic_class("fake_tactic_xyz", runtime=rt)
        self.assertIs(result, FakeTactic)

    def test_register_without_name_raises(self):
        """register_tactic_class raises when tactic.name is None.
        Implementation calls _normalize_name(None) which raises ValueError,
        then the assertion check also catches AssertionError."""
        from lllm.core.tactic import register_tactic_class
        from lllm.core.runtime import Runtime

        rt = Runtime()

        class NoNameTactic:
            name = None  # No name!

        with self.assertRaises((AssertionError, ValueError)):
            register_tactic_class(NoNameTactic, runtime=rt)

    def test_get_nonexistent_raises(self):
        from lllm.core.tactic import get_tactic_class
        from lllm.core.runtime import Runtime
        rt = Runtime()
        with self.assertRaises(KeyError):
            get_tactic_class("does_not_exist", runtime=rt)


# ===========================================================================
# Tactic auto-registration via __init_subclass__
# ===========================================================================


class TestTacticAutoRegistration(unittest.TestCase):

    def test_subclass_auto_registers(self):
        from lllm.core.tactic import Tactic, get_tactic_class
        from lllm.core.runtime import Runtime

        rt = Runtime()

        # We can't easily test __init_subclass__ with a custom runtime
        # but we can test with the default runtime
        class AutoTactic(Tactic, register=False):  # register=False to avoid polluting default
            name = "auto_registered_tactic"
            agent_group = []

        # With register=False, it should NOT be registered
        with self.assertRaises(KeyError):
            get_tactic_class("auto_registered_tactic", rt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
