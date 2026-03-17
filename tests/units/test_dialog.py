"""
Tests for lllm/core/dialog.py
Covers: TokenLogprob, Message, DialogTreeNode, Dialog, ContextManager.
"""
import base64
import copy
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_message(role=None, content="hello", name="user", **kwargs):
    from lllm.core.dialog import Message
    from lllm.core.const import Roles
    return Message(role=role or Roles.USER, content=content, name=name, **kwargs)


def _make_dialog(owner="agent", runtime=None):
    from lllm.core.dialog import Dialog
    from lllm.core.runtime import Runtime
    rt = runtime or Runtime()
    return Dialog(owner=owner, runtime=rt)


# ===========================================================================
# TokenLogprob
# ===========================================================================


class TestTokenLogprob(unittest.TestCase):

    def test_basic_construction(self):
        from lllm.core.dialog import TokenLogprob
        t = TokenLogprob(token="hello", logprob=-0.5)
        self.assertEqual(t.token, "hello")
        self.assertAlmostEqual(t.logprob, -0.5)

    def test_defaults(self):
        from lllm.core.dialog import TokenLogprob
        t = TokenLogprob()
        self.assertIsNone(t.token)
        self.assertIsNone(t.logprob)
        self.assertIsNone(t.bytes)
        self.assertEqual(t.top_logprobs, [])

    def test_nested_top_logprobs(self):
        from lllm.core.dialog import TokenLogprob
        t = TokenLogprob(
            token="a",
            logprob=-1.0,
            top_logprobs=[TokenLogprob(token="b", logprob=-2.0)],
        )
        self.assertEqual(len(t.top_logprobs), 1)
        self.assertEqual(t.top_logprobs[0].token, "b")

    def test_extra_fields_allowed(self):
        from lllm.core.dialog import TokenLogprob
        t = TokenLogprob(token="x", custom_field="extra")
        self.assertEqual(t.custom_field, "extra")


# ===========================================================================
# Message
# ===========================================================================


class TestMessage(unittest.TestCase):

    def test_basic_construction(self):
        from lllm.core.const import Roles, Modalities
        msg = _make_message()
        self.assertEqual(msg.role, Roles.USER)
        self.assertEqual(msg.content, "hello")
        self.assertEqual(msg.name, "user")
        self.assertEqual(msg.modality, Modalities.TEXT)
        self.assertFalse(msg.is_function_call)

    def test_is_function_call_false(self):
        msg = _make_message()
        self.assertFalse(msg.is_function_call)

    def test_is_function_call_true(self):
        from lllm.core.const import Roles, FunctionCall
        fc = FunctionCall(id="c1", name="fn", arguments={})
        msg = _make_message(role=Roles.ASSISTANT, function_calls=[fc])
        self.assertTrue(msg.is_function_call)

    def test_sanitized_name_clean(self):
        msg = _make_message(name="agent_007")
        self.assertEqual(msg.sanitized_name, "agent_007")

    def test_sanitized_name_replaces_special_chars(self):
        msg = _make_message(name="agent/007:test")
        # special chars become underscores
        sanitized = msg.sanitized_name
        self.assertNotIn("/", sanitized)
        self.assertNotIn(":", sanitized)

    def test_sanitized_name_truncated_at_64(self):
        long_name = "a" * 100
        msg = _make_message(name=long_name)
        self.assertLessEqual(len(msg.sanitized_name), 64)

    def test_cost_no_usage(self):
        from lllm.core.const import InvokeCost
        msg = _make_message()
        cost = msg.cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.total_tokens, 0)

    def test_cost_with_usage(self):
        msg = _make_message(usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cost": 0.001,
            "completion_cost": 0.002,
            "response_cost": 0.003,
        })
        cost = msg.cost
        self.assertEqual(cost.prompt_tokens, 100)
        self.assertEqual(cost.completion_tokens, 50)
        self.assertEqual(cost.total_tokens, 150)
        self.assertAlmostEqual(cost.cost, 0.003)

    def test_cost_with_details(self):
        msg = _make_message(usage={
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 50, "audio_tokens": 10},
            "completion_tokens_details": {"reasoning_tokens": 20, "audio_tokens": 5},
        })
        cost = msg.cost
        self.assertEqual(cost.cached_prompt_tokens, 50)
        self.assertEqual(cost.audio_prompt_tokens, 10)
        self.assertEqual(cost.reasoning_tokens, 20)
        self.assertEqual(cost.audio_completion_tokens, 5)

    def test_cost_inferred_total_tokens(self):
        """total_tokens should be inferred as prompt + completion if not provided."""
        msg = _make_message(usage={"prompt_tokens": 80, "completion_tokens": 40})
        self.assertEqual(msg.cost.total_tokens, 120)

    def test_to_dict_from_dict_roundtrip(self):
        from lllm.core.const import Roles
        msg = _make_message(
            role=Roles.ASSISTANT,
            content="Test reply",
            name="bot",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        d = msg.to_dict()
        restored = _make_message.__func__ if False else None
        from lllm.core.dialog import Message
        restored = Message.from_dict(d)
        self.assertEqual(restored.role, msg.role)
        self.assertEqual(restored.content, msg.content)
        self.assertEqual(restored.name, msg.name)

    def test_logprobs_coercion_from_dict(self):
        from lllm.core.dialog import Message, TokenLogprob
        from lllm.core.const import Roles
        msg = Message(
            role=Roles.ASSISTANT,
            content="hi",
            name="agent",
            logprobs=[
                {"token": "hello", "logprob": -0.1},
                0.5,
                "raw_token",
            ],
        )
        self.assertEqual(len(msg.logprobs), 3)
        self.assertIsInstance(msg.logprobs[0], TokenLogprob)
        self.assertEqual(msg.logprobs[0].token, "hello")
        # numeric becomes logprob
        self.assertAlmostEqual(msg.logprobs[1].logprob, 0.5)
        # string becomes token
        self.assertEqual(msg.logprobs[2].token, "raw_token")

    def test_logprobs_empty(self):
        from lllm.core.dialog import Message
        from lllm.core.const import Roles
        msg = Message(role=Roles.USER, content="hi", name="u", logprobs=None)
        self.assertEqual(msg.logprobs, [])

    def test_content_as_list(self):
        """Content can be a list of content parts (for multimodal)."""
        from lllm.core.const import Roles
        content = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": "data:..."}]
        msg = _make_message(role=Roles.USER, content=content)
        self.assertIsInstance(msg.content, list)


# ===========================================================================
# DialogTreeNode
# ===========================================================================


class TestDialogTreeNode(unittest.TestCase):

    def test_is_root_true(self):
        from lllm.core.dialog import DialogTreeNode
        node = DialogTreeNode()
        self.assertTrue(node.is_root)
        self.assertIsNone(node.parent_id)

    def test_depth_root_is_zero(self):
        from lllm.core.dialog import DialogTreeNode
        node = DialogTreeNode()
        self.assertEqual(node.depth, 0)

    def test_add_child_wires_refs(self):
        from lllm.core.dialog import DialogTreeNode
        parent = DialogTreeNode(owner="agent")
        child = DialogTreeNode(owner="agent")
        parent.add_child(child)

        self.assertEqual(child.parent_id, parent.dialog_id)
        self.assertIn(child.dialog_id, parent.children_ids)
        self.assertIn(child, parent._children)
        self.assertIs(child._parent, parent)

    def test_depth_child_is_one(self):
        from lllm.core.dialog import DialogTreeNode
        parent = DialogTreeNode()
        child = DialogTreeNode()
        parent.add_child(child)
        self.assertEqual(child.depth, 1)

    def test_depth_grandchild_is_two(self):
        from lllm.core.dialog import DialogTreeNode
        g = DialogTreeNode()
        p = DialogTreeNode()
        c = DialogTreeNode()
        g.add_child(p)
        p.add_child(c)
        self.assertEqual(c.depth, 2)

    def test_subtree_ids_single(self):
        from lllm.core.dialog import DialogTreeNode
        node = DialogTreeNode()
        ids = node.subtree_ids()
        self.assertEqual(ids, [node.dialog_id])

    def test_subtree_ids_with_children(self):
        from lllm.core.dialog import DialogTreeNode
        root = DialogTreeNode()
        c1 = DialogTreeNode()
        c2 = DialogTreeNode()
        gc = DialogTreeNode()
        root.add_child(c1)
        root.add_child(c2)
        c1.add_child(gc)
        ids = root.subtree_ids()
        self.assertIn(root.dialog_id, ids)
        self.assertIn(c1.dialog_id, ids)
        self.assertIn(c2.dialog_id, ids)
        self.assertIn(gc.dialog_id, ids)
        self.assertEqual(len(ids), 4)

    def test_to_dict_from_dict_roundtrip(self):
        from lllm.core.dialog import DialogTreeNode
        parent = DialogTreeNode(owner="agent", split_point=3)
        child = DialogTreeNode(owner="agent", last_n=5, first_k=1)
        parent.add_child(child)

        d = child.to_dict()
        self.assertEqual(d["parent_id"], parent.dialog_id)
        self.assertEqual(d["last_n"], 5)
        self.assertEqual(d["first_k"], 1)

        restored = DialogTreeNode.from_dict(d)
        self.assertEqual(restored.dialog_id, child.dialog_id)
        self.assertEqual(restored.parent_id, parent.dialog_id)
        self.assertEqual(restored.last_n, 5)

    def test_multiple_children_all_recorded(self):
        from lllm.core.dialog import DialogTreeNode
        parent = DialogTreeNode()
        for _ in range(5):
            child = DialogTreeNode()
            parent.add_child(child)
        self.assertEqual(len(parent.children_ids), 5)
        self.assertEqual(len(parent._children), 5)


# ===========================================================================
# Dialog
# ===========================================================================


class TestDialog(unittest.TestCase):

    def test_construction_defaults(self):
        d = _make_dialog()
        self.assertIsNotNone(d.session_name)
        self.assertIsNotNone(d.tree_node)
        self.assertTrue(d.is_root)
        self.assertEqual(d.depth, 0)
        self.assertEqual(len(d.messages), 0)
        self.assertIsNone(d.tail)
        self.assertIsNone(d.head)

    def test_append_message(self):
        from lllm.core.const import Roles
        d = _make_dialog()
        msg = _make_message(role=Roles.USER, content="hello")
        d.append(msg)
        self.assertEqual(len(d.messages), 1)
        self.assertIs(d.tail, msg)
        self.assertIs(d.head, msg)

    def test_append_sets_dialog_id(self):
        d = _make_dialog()
        msg = _make_message()
        d.append(msg)
        self.assertEqual(msg.metadata["dialog_id"], d.dialog_id)

    def test_put_text(self):
        from lllm.core.const import Roles
        d = _make_dialog()
        msg = d.put_text("Hello world!", name="user")
        self.assertEqual(msg.content, "Hello world!")
        self.assertEqual(msg.role, Roles.USER)
        self.assertIs(d.tail, msg)

    def test_put_text_custom_role(self):
        from lllm.core.const import Roles
        d = _make_dialog()
        msg = d.put_text("System msg", role=Roles.SYSTEM, name="system")
        self.assertEqual(msg.role, Roles.SYSTEM)

    def test_put_text_updates_top_prompt(self):
        d = _make_dialog()
        d.put_text("What is the weather?")
        self.assertIsNotNone(d.top_prompt)

    def test_put_prompt_with_prompt_object(self):
        from lllm.core.prompt import Prompt
        d = _make_dialog()
        p = Prompt(path="test", prompt="Hello {name}!")
        msg = d.put_prompt(p, prompt_args={"name": "Alice"})
        self.assertEqual(msg.content, "Hello Alice!")
        self.assertIs(d.top_prompt, p)

    def test_put_prompt_no_args_for_static(self):
        from lllm.core.prompt import Prompt
        d = _make_dialog()
        p = Prompt(path="static", prompt="Static prompt with no vars")
        msg = d.put_prompt(p)
        self.assertEqual(msg.content, "Static prompt with no vars")

    def test_put_image_from_base64(self):
        from lllm.core.const import Modalities
        # Create a valid PNG base64 string
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 20
        b64 = base64.b64encode(png_bytes).decode()
        d = _make_dialog()
        msg = d.put_image(b64, caption="test image")
        self.assertEqual(msg.modality, Modalities.IMAGE)
        self.assertEqual(msg.metadata.get("caption"), "test image")

    def test_put_image_from_pil(self):
        from PIL import Image
        from lllm.core.const import Modalities
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        d = _make_dialog()
        msg = d.put_image(img)
        self.assertEqual(msg.modality, Modalities.IMAGE)

    def test_put_image_invalid_base64_raises(self):
        d = _make_dialog()
        with self.assertRaises(ValueError):
            d.put_image("not_base64!!!")

    def test_put_image_valid_base64_non_image_raises(self):
        # Valid base64 but not an image
        bad_b64 = base64.b64encode(b"hello world").decode()
        d = _make_dialog()
        with self.assertRaises(ValueError):
            d.put_image(bad_b64)

    def test_put_image_invalid_type_raises(self):
        d = _make_dialog()
        with self.assertRaises(ValueError):
            d.put_image(12345)  # type: ignore

    def test_dialog_id_is_stable(self):
        d = _make_dialog()
        id1 = d.dialog_id
        id2 = d.dialog_id
        self.assertEqual(id1, id2)

    def test_cost_empty_dialog(self):
        from lllm.core.const import InvokeCost
        d = _make_dialog()
        cost = d.cost
        self.assertIsInstance(cost, InvokeCost)
        self.assertEqual(cost.total_tokens, 0)

    def test_cost_aggregates_messages(self):
        d = _make_dialog()
        d.put_text("Q1")
        d._messages[-1].usage = {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        d.put_text("Q2")
        d._messages[-1].usage = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
        cost = d.cost
        self.assertEqual(cost.prompt_tokens, 150)
        self.assertEqual(cost.total_tokens, 180)

    # -----------------------------------------------------------------------
    # Fork tests
    # -----------------------------------------------------------------------

    def test_fork_basic(self):
        d = _make_dialog()
        d.put_text("msg1")
        d.put_text("msg2")
        d.put_text("msg3")
        child = d.fork()

        self.assertEqual(len(child.messages), 3)
        self.assertFalse(child.is_root)
        self.assertEqual(child.depth, 1)
        self.assertIs(child.parent, d)
        self.assertIn(child, d.children)

    def test_fork_tree_node_linked(self):
        d = _make_dialog()
        d.put_text("msg1")
        d.put_text("msg2")
        child = d.fork()

        self.assertEqual(child.tree_node.parent_id, d.tree_node.dialog_id)
        self.assertIn(child.tree_node.dialog_id, d.tree_node.children_ids)

    def test_fork_deep_copies_messages(self):
        d = _make_dialog()
        d.put_text("original content")
        child = d.fork()

        # Modify original — child should not be affected
        d._messages[0].metadata["dirty"] = True
        self.assertNotIn("dirty", child._messages[0].metadata)

    def test_fork_with_last_n(self):
        d = _make_dialog()
        for i in range(10):
            d.put_text(f"msg_{i}")
        child = d.fork(last_n=3, first_k=1)
        # first_k=1 system prompt + last 3 messages
        self.assertEqual(len(child.messages), 4)
        self.assertEqual(child.messages[-1].content, "msg_9")
        self.assertEqual(child.messages[-2].content, "msg_8")
        self.assertEqual(child.messages[-3].content, "msg_7")
        self.assertEqual(child.messages[0].content, "msg_0")

    def test_fork_last_n_exceeds_length_resets(self):
        """last_n >= len should be treated as 0 (take all messages)."""
        d = _make_dialog()
        d.put_text("only")
        child = d.fork(last_n=100)
        self.assertEqual(len(child.messages), 1)

    def test_fork_split_point_recorded(self):
        d = _make_dialog()
        d.put_text("a")
        d.put_text("b")
        child = d.fork()
        self.assertEqual(child.tree_node.split_point, 2)

    def test_fork_inherits_runtime_and_prompt(self):
        from lllm.core.prompt import Prompt
        d = _make_dialog()
        p = Prompt(path="sys", prompt="System prompt")
        d.top_prompt = p
        child = d.fork()
        self.assertIs(child.top_prompt, p)
        self.assertIs(child.runtime, d.runtime)

    def test_multiple_forks(self):
        d = _make_dialog()
        d.put_text("base")
        c1 = d.fork()
        c2 = d.fork()
        self.assertEqual(len(d.children), 2)
        ids = d.tree_node.subtree_ids()
        self.assertIn(c1.dialog_id, ids)
        self.assertIn(c2.dialog_id, ids)

    def test_nested_fork_depth(self):
        d = _make_dialog()
        d.put_text("base")
        c1 = d.fork()
        c1.put_text("child")
        c2 = c1.fork()
        self.assertEqual(c2.depth, 2)
        self.assertIs(c2.parent, c1)

    # -----------------------------------------------------------------------
    # Overview / tree_overview display
    # -----------------------------------------------------------------------

    def test_overview_returns_string(self):
        d = _make_dialog()
        d.put_text("Hello!")
        result = d.overview()
        self.assertIsInstance(result, str)
        self.assertIn("Hello!", result)

    def test_overview_max_length_truncates(self):
        d = _make_dialog()
        d.put_text("A" * 200)
        result = d.overview(max_length=50)
        self.assertIn("...", result)

    def test_overview_remove_tail_skips_last(self):
        d = _make_dialog()
        d.put_text("first")
        d.put_text("last")
        result = d.overview(remove_tail=True)
        self.assertIn("first", result)
        self.assertNotIn("last", result)

    def test_tree_overview_structure(self):
        d = _make_dialog(owner="root")
        d.put_text("base")
        c = d.fork()
        tree = d.tree_overview()
        self.assertIn("owner=root", tree)
        # child is indented
        lines = tree.split("\n")
        self.assertGreaterEqual(len(lines), 2)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def test_to_dict_from_dict(self):
        from lllm.core.runtime import Runtime
        rt = Runtime()
        from lllm.core.prompt import Prompt
        p = Prompt(path="sys", prompt="System")
        rt.register_prompt(p, namespace="test.prompts")

        d = _make_dialog(runtime=rt)
        d.top_prompt = p
        d.put_text("Hello")
        d.put_text("World")

        data = d.to_dict()
        self.assertIn("messages", data)
        self.assertIn("session_name", data)
        self.assertIn("tree_node", data)
        self.assertEqual(len(data["messages"]), 2)

    def test_properties(self):
        d = _make_dialog()
        d.put_text("first")
        d.put_text("last")
        self.assertEqual(d.head.content, "first")
        self.assertEqual(d.tail.content, "last")
        self.assertEqual(len(d.messages), 2)
        self.assertEqual(d.children, [])
        self.assertIsNone(d.parent)


# ===========================================================================
# ContextManager (abstract base)
# ===========================================================================


class TestContextManager(unittest.TestCase):

    def test_abstract_cannot_instantiate(self):
        from lllm.core.dialog import ContextManager
        with self.assertRaises(TypeError):
            ContextManager()  # type: ignore

    def test_concrete_subclass_passthrough(self):
        from lllm.core.dialog import ContextManager, Dialog

        class PassthroughManager(ContextManager):
            name = "passthrough"

            def __call__(self, dialog: Dialog) -> Dialog:
                return dialog

        d = _make_dialog()
        mgr = PassthroughManager()
        result = mgr(d)
        self.assertIs(result, d)

    def test_name_attribute_on_default(self):
        from lllm.core.dialog import DefaultContextManager
        self.assertEqual(DefaultContextManager.name, "default")

    def test_runtime_register_uses_class_name(self):
        from lllm.core.dialog import ContextManager, Dialog
        from lllm.core.runtime import Runtime

        class MyManager(ContextManager):
            name = "my_manager"

            def __call__(self, dialog: Dialog) -> Dialog:
                return dialog

        rt = Runtime()
        rt.register_context_manager(MyManager)
        retrieved = rt.get_context_manager("my_manager")
        self.assertIs(retrieved, MyManager)

    def test_runtime_register_requires_name(self):
        from lllm.core.dialog import ContextManager, Dialog
        from lllm.core.runtime import Runtime

        class Unnamed(ContextManager):
            # no name class attribute
            def __call__(self, dialog: Dialog) -> Dialog:
                return dialog

        rt = Runtime()
        with self.assertRaises(ValueError):
            rt.register_context_manager(Unnamed)

    def test_subclass_without_call_raises_on_instantiate(self):
        from lllm.core.dialog import ContextManager

        class Incomplete(ContextManager):
            name = "incomplete"
            # missing __call__

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore


# ===========================================================================
# DefaultContextManager
# ===========================================================================


def _make_filled_dialog(n_messages: int, content_per_msg: str = "hello world") -> 'Dialog':
    """Helper: dialog with a system prompt + n_messages user messages."""
    from lllm.core.dialog import Dialog
    from lllm.core.const import Roles
    from lllm.core.runtime import Runtime

    d = Dialog(owner="agent", runtime=Runtime())
    d.put_text("You are a helpful assistant.", role=Roles.SYSTEM, name="system")
    for i in range(n_messages):
        d.put_text(f"{content_per_msg} {i}", name="user")
    return d


class TestDefaultContextManager(unittest.TestCase):

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_name_is_default(self):
        from lllm.core.dialog import DefaultContextManager
        self.assertEqual(DefaultContextManager.name, "default")

    def test_init_stores_model_and_max_tokens(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o", max_tokens=8192)
        self.assertEqual(cm.model_name, "gpt-4o")
        self.assertEqual(cm.max_tokens, 8192)

    def test_max_tokens_from_litellm_when_not_set(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o")
        # Should return a positive integer (auto-detected or 4096 fallback)
        self.assertIsInstance(cm.max_tokens, int)
        self.assertGreater(cm.max_tokens, 0)

    def test_max_tokens_fallback_on_unknown_model(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("totally-unknown-model-xyz")
        # Should not raise; returns a sensible fallback
        self.assertIsInstance(cm.max_tokens, int)
        self.assertGreater(cm.max_tokens, 0)

    # ------------------------------------------------------------------
    # No-op when within limit
    # ------------------------------------------------------------------

    def test_returns_same_dialog_when_within_limit(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o", max_tokens=128000)
        d = _make_filled_dialog(5)
        result = cm(d)
        # Small dialog fits easily — same object returned
        self.assertIs(result, d)

    def test_empty_dialog_returned_unchanged(self):
        from lllm.core.dialog import DefaultContextManager, Dialog
        from lllm.core.runtime import Runtime
        cm = DefaultContextManager("gpt-4o", max_tokens=1000)
        d = Dialog(owner="agent", runtime=Runtime())
        result = cm(d)
        self.assertIs(result, d)

    # ------------------------------------------------------------------
    # Truncation behaviour
    # ------------------------------------------------------------------

    def test_truncation_preserves_first_message(self):
        from lllm.core.dialog import DefaultContextManager
        # Very tight limit forces truncation
        cm = DefaultContextManager("gpt-4o-mini", max_tokens=200)
        d = _make_filled_dialog(20, content_per_msg="word " * 30)
        result = cm(d)
        # First message (system prompt) must always survive
        self.assertEqual(result.messages[0].content, d.messages[0].content)

    def test_truncation_keeps_most_recent_messages(self):
        from lllm.core.dialog import DefaultContextManager
        # Use a limit generous enough to fit a few messages but not all 20
        cm = DefaultContextManager("gpt-4o-mini", max_tokens=600)
        d = _make_filled_dialog(20, content_per_msg="filler " * 10)
        result = cm(d)
        if len(result.messages) < len(d.messages) and len(result.messages) > 1:
            # Most-recent messages should be retained; the very last one must survive
            self.assertEqual(result.messages[-1].content, d.messages[-1].content)

    def test_truncation_returns_child_dialog(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o-mini", max_tokens=200)
        d = _make_filled_dialog(30, content_per_msg="x " * 20)
        result = cm(d)
        if result is not d:
            # Must be wired into the tree as a child
            self.assertFalse(result.is_root)
            self.assertIs(result.parent, d)
            self.assertIn(result, d.children)

    def test_truncated_dialog_has_fewer_messages(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o-mini", max_tokens=200)
        d = _make_filled_dialog(50, content_per_msg="token " * 20)
        result = cm(d)
        self.assertLessEqual(len(result.messages), len(d.messages))

    def test_border_message_gets_truncation_prefix(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o-mini", max_tokens=200)
        d = _make_filled_dialog(50, content_per_msg="token " * 20)
        result = cm(d)
        # If truncation happened and there are non-system messages,
        # the oldest non-system kept message may carry the truncation prefix.
        if len(result.messages) < len(d.messages) and len(result.messages) > 1:
            # At least one message should either be intact or carry the prefix
            non_system = [m for m in result.messages if m.name != "system"]
            if non_system:
                first_kept = non_system[0]
                # Either full content or truncated with prefix
                is_truncated = "[...earlier content truncated...]" in str(first_kept.content)
                is_full = first_kept.content in [m.content for m in d.messages]
                self.assertTrue(is_truncated or is_full)

    def test_to_raw_produces_correct_format(self):
        from lllm.core.dialog import DefaultContextManager
        from lllm.core.const import Roles
        cm = DefaultContextManager("gpt-4o")
        from lllm.core.dialog import Message
        msg = Message(role=Roles.USER, content="hello", name="user")
        raw = cm._to_raw(msg)
        self.assertIn("role", raw)
        self.assertIn("content", raw)
        self.assertEqual(raw["role"], "user")
        self.assertEqual(raw["content"], "hello")

    def test_count_returns_positive_int(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o")
        count = cm._count([{"role": "user", "content": "hello world"}])
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    # ------------------------------------------------------------------
    # Safety buffer
    # ------------------------------------------------------------------

    def test_safety_buffer_default_is_5000(self):
        from lllm.core.dialog import DefaultContextManager
        self.assertEqual(DefaultContextManager.SAFETY_BUFFER, 5000)

    def test_effective_limit_respects_safety_buffer(self):
        from lllm.core.dialog import DefaultContextManager
        cm = DefaultContextManager("gpt-4o", max_tokens=10000)
        effective = cm.max_tokens - cm.SAFETY_BUFFER
        self.assertEqual(effective, 5000)

    # ------------------------------------------------------------------
    # ContextManagerConfig integration
    # ------------------------------------------------------------------

    def test_config_build_default_type(self):
        from lllm.core.config import ContextManagerConfig
        from lllm.core.dialog import DefaultContextManager
        from lllm.core.runtime import Runtime
        cfg = ContextManagerConfig(type="default", max_tokens=16000)
        cm = cfg.build("gpt-4o", Runtime())
        self.assertIsInstance(cm, DefaultContextManager)
        self.assertEqual(cm.model_name, "gpt-4o")
        self.assertEqual(cm.max_tokens, 16000)

    def test_config_build_null_type_returns_none(self):
        from lllm.core.config import ContextManagerConfig
        from lllm.core.runtime import Runtime
        for null_val in ("null", "none", None):
            cfg = ContextManagerConfig(type=null_val)
            cm = cfg.build("gpt-4o", Runtime())
            self.assertIsNone(cm, f"Expected None for type={null_val!r}")

    def test_config_build_custom_type_from_runtime(self):
        from lllm.core.config import ContextManagerConfig
        from lllm.core.dialog import ContextManager, Dialog
        from lllm.core.runtime import Runtime

        class CustomManager(ContextManager):
            name = "custom"

            def __init__(self, model_name: str, max_tokens=None):
                self.model_name = model_name

            def __call__(self, dialog: Dialog) -> Dialog:
                return dialog

        rt = Runtime()
        rt.register_context_manager(CustomManager)
        cfg = ContextManagerConfig(type="custom")
        cm = cfg.build("gpt-4o", rt)
        self.assertIsInstance(cm, CustomManager)
        self.assertEqual(cm.model_name, "gpt-4o")

    def test_config_from_dict(self):
        from lllm.core.config import ContextManagerConfig
        cfg = ContextManagerConfig.from_dict({"type": "default", "max_tokens": 32000})
        self.assertEqual(cfg.type, "default")
        self.assertEqual(cfg.max_tokens, 32000)

    def test_config_from_dict_defaults(self):
        from lllm.core.config import ContextManagerConfig
        cfg = ContextManagerConfig.from_dict({})
        self.assertEqual(cfg.type, "default")
        self.assertIsNone(cfg.max_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
