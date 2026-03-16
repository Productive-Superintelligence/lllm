"""
Tests for the LLLM package system.
"""
import os
import sys
import textwrap
import tempfile
import unittest
import logging
from pathlib import Path

sys.path.insert(0, "/home/claude")


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


# ===========================================================================
# ResourceNode
# ===========================================================================

class TestResourceNode(unittest.TestCase):

    def test_eager(self):
        from lllm.core.resource import ResourceNode
        n = ResourceNode.eager("k", "v", namespace="pkg.prompts")
        self.assertEqual(n.qualified_key, "pkg.prompts:k")
        self.assertEqual(n.value, "v")
        self.assertTrue(n.is_loaded)

    def test_lazy(self):
        from lllm.core.resource import ResourceNode
        calls = 0
        def loader():
            nonlocal calls; calls += 1; return 42
        n = ResourceNode.lazy("k", loader, namespace="pkg.configs")
        self.assertFalse(n.is_loaded)
        self.assertEqual(n.value, 42)
        self.assertEqual(n.value, 42)  # cached
        self.assertEqual(calls, 1)

    def test_no_namespace(self):
        from lllm.core.resource import ResourceNode
        self.assertEqual(ResourceNode.eager("k", 1).qualified_key, "k")


class TestPackageInfo(unittest.TestCase):

    def test_effective_name(self):
        from lllm.core.resource import PackageInfo
        self.assertEqual(PackageInfo(name="x").effective_name, "x")
        self.assertEqual(PackageInfo(name="x", alias="y").effective_name, "y")


# ===========================================================================
# Runtime
# ===========================================================================

class TestRuntime(unittest.TestCase):

    def setUp(self):
        from lllm.core.runtime import Runtime
        self.rt = Runtime()

    def test_register_and_get(self):
        from lllm.core.resource import ResourceNode
        n = ResourceNode.eager("res", "val", namespace="pkg.prompts", resource_type="prompt")
        self.rt.register(n)
        self.assertEqual(self.rt.get("pkg.prompts:res"), "val")

    def test_default_namespace_resolution(self):
        from lllm.core.resource import ResourceNode
        self.rt._default_namespace = "mypkg"
        self.rt.register(ResourceNode.eager("foo", 1, namespace="mypkg.prompts", resource_type="prompt"))
        # Bare key resolves via default namespace
        self.assertEqual(self.rt.get("foo", resource_type="prompt"), 1)

    def test_pkg_colon_shorthand(self):
        """``get_prompt("pkg:key")`` inserts ``.prompts``."""
        from lllm.core.prompt import Prompt
        p = Prompt(path="sys", prompt="hi")
        self.rt.register_prompt(p, namespace="mypkg.prompts")
        self.assertIs(self.rt.get_prompt("mypkg:sys"), p)

    def test_not_found_raises(self):
        with self.assertRaises(KeyError):
            self.rt.get("nonexistent")

    def test_type_mismatch_raises(self):
        from lllm.core.resource import ResourceNode
        self.rt.register(ResourceNode.eager("x", 1, resource_type="prompt"))
        with self.assertRaises(TypeError):
            self.rt.get("x", resource_type="tactic")

    def test_has(self):
        from lllm.core.resource import ResourceNode
        self.rt.register(ResourceNode.eager("x", 1))
        self.assertTrue(self.rt.has("x"))
        self.assertFalse(self.rt.has("y"))

    def test_keys_filtered(self):
        from lllm.core.resource import ResourceNode
        self.rt.register(ResourceNode.eager("a", 1, resource_type="prompt"))
        self.rt.register(ResourceNode.eager("b", 2, resource_type="tactic"))
        self.assertIn("a", self.rt.keys("prompt"))
        self.assertNotIn("b", self.rt.keys("prompt"))

    def test_reset(self):
        from lllm.core.resource import ResourceNode, PackageInfo
        self.rt.register(ResourceNode.eager("k", 1))
        self.rt.register_package(PackageInfo(name="p"))
        self.rt._default_namespace = "p"
        self.rt._discovery_done = True
        self.rt.reset()
        self.assertEqual(len(self.rt._resources), 0)
        self.assertIsNone(self.rt._default_namespace)
        self.assertFalse(self.rt._discovery_done)


class TestNamedRuntimes(unittest.TestCase):

    def test_get_nonexistent(self):
        from lllm.core.runtime import get_runtime
        with self.assertRaises(KeyError):
            get_runtime("nonexistent_xyz")


# ===========================================================================
# TOML entry parsing
# ===========================================================================

class TestParsing(unittest.TestCase):

    def test_path_plain(self):
        from lllm.core.config import _parse_path_entry
        self.assertEqual(_parse_path_entry("./p").path, "./p")
        self.assertIsNone(_parse_path_entry("./p").prefix)

    def test_path_under_keyword(self):
        from lllm.core.config import _parse_path_entry
        p = _parse_path_entry("./p under vf")
        self.assertEqual(p.path, "./p")
        self.assertEqual(p.prefix, "vf")

    def test_path_table_prefix(self):
        from lllm.core.config import _parse_path_entry
        self.assertEqual(_parse_path_entry({"path": "./p", "prefix": "vf"}).prefix, "vf")

    def test_path_table_under_key(self):
        from lllm.core.config import _parse_path_entry
        self.assertEqual(_parse_path_entry({"path": "./p", "under": "vf"}).prefix, "vf")

    def test_dep_plain(self):
        from lllm.core.config import _parse_dependency_entry
        self.assertIsNone(_parse_dependency_entry("./pkg").alias)

    def test_dep_as_keyword(self):
        from lllm.core.config import _parse_dependency_entry
        self.assertEqual(_parse_dependency_entry("./pkg as p1").alias, "p1")

    def test_dep_table_alias(self):
        from lllm.core.config import _parse_dependency_entry
        self.assertEqual(_parse_dependency_entry({"path": "./pkg", "alias": "p1"}).alias, "p1")

    def test_dep_table_as_key(self):
        from lllm.core.config import _parse_dependency_entry
        self.assertEqual(_parse_dependency_entry({"path": "./pkg", "as": "p1"}).alias, "p1")


# ===========================================================================
# load_package integration
# ===========================================================================

class TestLoadPackage(unittest.TestCase):

    def setUp(self):
        from lllm.core.runtime import Runtime
        self.rt = Runtime()

    def test_single_package_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write(tmp / "lllm.toml", '[package]\nname = "tp"\n')
            _write(tmp / "prompts" / "greet.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                hello = Prompt(path="hello", prompt="Hello {name}!")
                bye = Prompt(path="bye", prompt="Bye {name}!")
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            self.assertEqual(self.rt._default_namespace, "tp")
            self.assertTrue(self.rt.has("tp.prompts:greet/hello"))
            self.assertTrue(self.rt.has("tp.prompts:greet/bye"))

            p = self.rt.get_prompt("tp:greet/hello")
            self.assertEqual(p.prompt, "Hello {name}!")

            # Bare key via default namespace
            p2 = self.rt.get_prompt("greet/hello")
            self.assertIs(p2, p)

    def test_configs_lazy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write(tmp / "lllm.toml", '[package]\nname = "tp"\n')
            _write(tmp / "configs" / "default.yaml", "model: gpt-4o\n")

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            node = self.rt.get_node("tp.configs:default")
            self.assertFalse(node.is_loaded)

            cfg = self.rt.get_config("default")
            self.assertEqual(cfg["model"], "gpt-4o")
            self.assertTrue(node.is_loaded)

    def test_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            child = tmp / "child"
            _write(child / "lllm.toml", '[package]\nname = "child"\n')
            _write(child / "prompts" / "a.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                x = Prompt(path="x", prompt="X")
            """))
            _write(tmp / "lllm.toml", textwrap.dedent("""\
                [package]
                name = "root"
                [dependencies]
                packages = ["./child"]
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            self.assertEqual(self.rt._default_namespace, "root")
            self.assertTrue(self.rt.has("child.prompts:a/x"))
            self.assertEqual(self.rt.get_prompt("child:a/x").prompt, "X")

    def test_dependency_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            child = tmp / "child"
            _write(child / "lllm.toml", '[package]\nname = "child"\n')
            _write(child / "prompts" / "u.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                h = Prompt(path="h", prompt="Help")
            """))
            _write(tmp / "lllm.toml", textwrap.dedent("""\
                [package]
                name = "root"
                [dependencies]
                packages = ["./child as c"]
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            # Original and alias both work
            self.assertTrue(self.rt.has("child.prompts:u/h"))
            self.assertTrue(self.rt.has("c.prompts:u/h"))
            self.assertEqual(self.rt.get("c.prompts:u/h").prompt, "Help")

    def test_reexport_under_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            child = tmp / "child"
            _write(child / "lllm.toml", '[package]\nname = "child"\n')
            _write(child / "prompts" / "t.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                s = Prompt(path="s", prompt="Search")
            """))
            child_prompts = str((child / "prompts").resolve())
            _write(tmp / "lllm.toml", textwrap.dedent(f"""\
                [package]
                name = "root"
                [prompts]
                paths = ["{child_prompts} under vendor"]
                [dependencies]
                packages = ["./child"]
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            # Root namespace has prefix
            self.assertTrue(self.rt.has("root.prompts:vendor/t/s"))
            # Child namespace has no prefix
            self.assertTrue(self.rt.has("child.prompts:t/s"))

    def test_cycle_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a", tmp / "b"
            _write(a / "lllm.toml", f'[package]\nname = "a"\n[dependencies]\npackages = ["{b.resolve()}"]\n')
            _write(b / "lllm.toml", f'[package]\nname = "b"\n[dependencies]\npackages = ["{a.resolve()}"]\n')

            from lllm.core.config import load_package
            load_package(str(a / "lllm.toml"), runtime=self.rt)  # should not hang
            self.assertIn("a", self.rt.packages)
            self.assertIn("b", self.rt.packages)

    def test_empty_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write(tmp / "lllm.toml", '[package]\nname = "empty"\n')

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)
            self.assertIn("empty", self.rt.packages)
            self.assertEqual(len(self.rt.keys("prompt")), 0)

    def test_nested_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write(tmp / "lllm.toml", '[package]\nname = "deep"\n')
            _write(tmp / "prompts" / "a" / "b" / "c.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                dp = Prompt(path="dp", prompt="deep")
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            self.assertTrue(self.rt.has("deep.prompts:a/b/c/dp"))

    def test_table_form_under(self):
        """TOML inline table form for paths with prefix."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompts_dir = tmp / "myprompts"
            _write(prompts_dir / "q.py", textwrap.dedent("""\
                from lllm.core.prompt import Prompt
                p = Prompt(path="p", prompt="P")
            """))
            # Use inline table: {path = "...", prefix = "..."}
            _write(tmp / "lllm.toml", textwrap.dedent(f"""\
                [package]
                name = "tp"
                [prompts]
                paths = [{{path = "{prompts_dir.resolve()}", prefix = "vf"}}]
            """))

            from lllm.core.config import load_package
            load_package(str(tmp / "lllm.toml"), runtime=self.rt)

            self.assertTrue(self.rt.has("tp.prompts:vf/q/p"))


# ===========================================================================
# Loader convenience API
# ===========================================================================

class TestLoader(unittest.TestCase):

    def setUp(self):
        from lllm.core.runtime import Runtime
        self.rt = Runtime()
        self.rt._default_namespace = "pkg"

    def test_load_resource_requires_colon(self):
        from lllm.core.resource import load_resource
        with self.assertRaises(ValueError):
            load_resource("no_colon", runtime=self.rt)

    def test_load_resource_section_only(self):
        from lllm.core.resource import ResourceNode
        from lllm.core.resource import load_resource
        self.rt.register(ResourceNode.eager("logo", b"png", namespace="pkg.assets",
                                            resource_type="assets"))
        self.assertEqual(load_resource("assets:logo", runtime=self.rt), b"png")

    def test_load_resource_full_url(self):
        from lllm.core.resource import ResourceNode
        from lllm.core.resource import load_resource
        self.rt.register(ResourceNode.eager("x", 99, namespace="other.data",
                                            resource_type="data"))
        self.assertEqual(load_resource("other.data:x", runtime=self.rt), 99)

    def test_load_prompt(self):
        from lllm.core.prompt import Prompt
        from lllm.core.resource import load_prompt
        p = Prompt(path="sys", prompt="hi")
        self.rt.register_prompt(p, namespace="pkg.prompts")
        self.assertIs(load_prompt("sys", runtime=self.rt), p)
        self.assertIs(load_prompt("pkg:sys", runtime=self.rt), p)


# ===========================================================================
# Config resolution (deep_merge + base inheritance)
# ===========================================================================

class TestDeepMerge(unittest.TestCase):

    def test_scalar_override(self):
        from lllm.core.config import _deep_merge
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        self.assertEqual(result, {"a": 1, "b": 3, "c": 4})

    def test_nested_dict_merge(self):
        from lllm.core.config import _deep_merge
        base = {"model_args": {"temperature": 0.1, "top_p": 0.9}}
        override = {"model_args": {"temperature": 0.5, "max_tokens": 100}}
        result = _deep_merge(base, override)
        self.assertEqual(result["model_args"], {
            "temperature": 0.5, "top_p": 0.9, "max_tokens": 100,
        })

    def test_list_replacement(self):
        from lllm.core.config import _deep_merge
        base = {"items": [1, 2]}
        override = {"items": [3]}
        self.assertEqual(_deep_merge(base, override)["items"], [3])

    def test_no_mutation(self):
        from lllm.core.config import _deep_merge
        base = {"nested": {"a": 1}}
        override = {"nested": {"b": 2}}
        result = _deep_merge(base, override)
        self.assertNotIn("b", base["nested"])
        self.assertEqual(result["nested"], {"a": 1, "b": 2})


class TestResolveConfig(unittest.TestCase):

    def setUp(self):
        from lllm.core.runtime import Runtime
        self.rt = Runtime()
        self.rt._default_namespace = "pkg"

    def test_no_base(self):
        from lllm.core.config import resolve_config
        self.rt.register_config("simple", {"model_name": "gpt-4o"},
                                namespace="pkg.configs")
        cfg = resolve_config("simple", self.rt)
        self.assertEqual(cfg["model_name"], "gpt-4o")

    def test_single_inheritance(self):
        from lllm.core.config import resolve_config
        self.rt.register_config("parent", {
            "global": {"model_name": "gpt-4o", "model_args": {"temperature": 0.1}},
            "agent_configs": [],
        }, namespace="pkg.configs")
        self.rt.register_config("child", {
            "base": "parent",
            "global": {"model_args": {"max_tokens": 500}},
            "agent_configs": [{"name": "coder", "system_prompt": "Code!"}],
        }, namespace="pkg.configs")

        cfg = resolve_config("child", self.rt)
        self.assertNotIn("base", cfg)
        # Global merged: parent's temperature + child's max_tokens
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o")
        self.assertEqual(cfg["global"]["model_args"]["temperature"], 0.1)
        self.assertEqual(cfg["global"]["model_args"]["max_tokens"], 500)
        # Child's agent_configs replaces parent's (list replacement)
        self.assertEqual(len(cfg["agent_configs"]), 1)

    def test_chain_inheritance(self):
        from lllm.core.config import resolve_config
        self.rt.register_config("grandparent", {"a": 1, "b": 1},
                                namespace="pkg.configs")
        self.rt.register_config("parent", {"base": "grandparent", "b": 2, "c": 2},
                                namespace="pkg.configs")
        self.rt.register_config("child", {"base": "parent", "c": 3, "d": 3},
                                namespace="pkg.configs")

        cfg = resolve_config("child", self.rt)
        self.assertEqual(cfg, {"a": 1, "b": 2, "c": 3, "d": 3})

    def test_circular_inheritance_raises(self):
        from lllm.core.config import resolve_config
        self.rt.register_config("x", {"base": "y"}, namespace="pkg.configs")
        self.rt.register_config("y", {"base": "x"}, namespace="pkg.configs")
        with self.assertRaises(ValueError, msg="Circular"):
            resolve_config("x", self.rt)


# ===========================================================================
# AgentSpec and parse_agent_configs
# ===========================================================================

class TestAgentSpec(unittest.TestCase):

    def test_from_config_with_path(self):
        from lllm.core.config import AgentSpec
        raw = {
            "name": "coder",
            "model_name": "gpt-4o",
            "system_prompt_path": "coding/system",
            "model_args": {"temperature": 0.2},
        }
        spec = AgentSpec.from_config("coder", raw)
        self.assertEqual(spec.name, "coder")
        self.assertEqual(spec.model, "gpt-4o")
        self.assertEqual(spec.system_prompt_path, "coding/system")
        self.assertIsNone(spec.system_prompt)
        self.assertEqual(spec.model_args, {"temperature": 0.2})

    def test_from_config_with_inline_prompt(self):
        from lllm.core.config import AgentSpec
        raw = {
            "name": "helper",
            "model_name": "gpt-4o-mini",
            "system_prompt": "You are a helpful assistant.",
        }
        spec = AgentSpec.from_config("helper", raw)
        self.assertIsNotNone(spec.system_prompt)
        self.assertEqual(spec.system_prompt.prompt, "You are a helpful assistant.")
        self.assertIsNone(spec.system_prompt_path)

    def test_from_config_missing_model_raises(self):
        from lllm.core.config import AgentSpec
        with self.assertRaises(ValueError, msg="model_name"):
            AgentSpec.from_config("bad", {"system_prompt": "hi"})

    def test_from_config_missing_prompt_raises(self):
        from lllm.core.config import AgentSpec
        with self.assertRaises(ValueError, msg="system_prompt"):
            AgentSpec.from_config("bad", {"model_name": "gpt-4o"})

    def test_from_config_extra_keys_become_model_args(self):
        from lllm.core.config import AgentSpec
        raw = {
            "model_name": "gpt-4o",
            "system_prompt": "Hi",
            "model_args": {"temperature": 0.1},
            "top_p": 0.9,  # unknown key → model_args
        }
        spec = AgentSpec.from_config("test", raw)
        self.assertEqual(spec.model_args["temperature"], 0.1)
        self.assertEqual(spec.model_args["top_p"], 0.9)

    def test_from_config_execution_limits(self):
        from lllm.core.config import AgentSpec
        raw = {
            "model_name": "gpt-4o",
            "system_prompt": "Hi",
            "max_exception_retry": 10,
            "max_interrupt_steps": 20,
            "max_llm_recall": 5,
        }
        spec = AgentSpec.from_config("test", raw)
        self.assertEqual(spec.max_exception_retry, 10)
        self.assertEqual(spec.max_interrupt_steps, 20)
        self.assertEqual(spec.max_llm_recall, 5)
        # These should NOT leak into model_args
        self.assertNotIn("max_exception_retry", spec.model_args)

    def test_from_config_extra_settings(self):
        from lllm.core.config import AgentSpec
        raw = {
            "model_name": "gpt-4o",
            "system_prompt": "Hi",
            "extra_settings": {"context_manager": "pruner"},
        }
        spec = AgentSpec.from_config("test", raw)
        self.assertEqual(spec.extra_settings, {"context_manager": "pruner"})


class TestParseAgentConfigs(unittest.TestCase):

    def test_basic(self):
        from lllm.core.config import parse_agent_configs
        config = {
            "agent_configs": [
                {"name": "coder", "model_name": "gpt-4o", "system_prompt": "Code!"},
                {"name": "reviewer", "model_name": "gpt-4o-mini", "system_prompt": "Review!"},
            ],
        }
        specs = parse_agent_configs(config, ["coder", "reviewer"], "test_tactic")
        self.assertEqual(set(specs.keys()), {"coder", "reviewer"})
        self.assertEqual(specs["coder"].model, "gpt-4o")
        self.assertEqual(specs["reviewer"].model, "gpt-4o-mini")

    def test_global_merge(self):
        from lllm.core.config import parse_agent_configs
        config = {
            "global": {
                "model_name": "gpt-4o",
                "model_args": {"temperature": 0.1},
                "max_exception_retry": 5,
            },
            "agent_configs": [
                {"name": "coder", "system_prompt": "Code!",
                 "model_args": {"max_tokens": 1000}},
            ],
        }
        specs = parse_agent_configs(config, ["coder"], "test")
        coder = specs["coder"]
        # model_name from global
        self.assertEqual(coder.model, "gpt-4o")
        # model_args merged: global temp + per-agent max_tokens
        self.assertEqual(coder.model_args["temperature"], 0.1)
        self.assertEqual(coder.model_args["max_tokens"], 1000)
        # max_exception_retry from global
        self.assertEqual(coder.max_exception_retry, 5)

    def test_agent_overrides_global(self):
        from lllm.core.config import parse_agent_configs
        config = {
            "global": {"model_name": "gpt-4o", "model_args": {"temperature": 0.1}},
            "agent_configs": [
                {"name": "hot", "system_prompt": "Be creative!",
                 "model_name": "gpt-4o-mini",
                 "model_args": {"temperature": 0.9}},
            ],
        }
        specs = parse_agent_configs(config, ["hot"], "test")
        self.assertEqual(specs["hot"].model, "gpt-4o-mini")
        self.assertEqual(specs["hot"].model_args["temperature"], 0.9)

    def test_missing_agent_raises(self):
        from lllm.core.config import parse_agent_configs
        config = {"agent_configs": [{"name": "a", "model_name": "x", "system_prompt": "y"}]}
        with self.assertRaises(ValueError, msg="not found"):
            parse_agent_configs(config, ["a", "b"], "test")

    def test_missing_name_raises(self):
        from lllm.core.config import parse_agent_configs
        config = {"agent_configs": [{"model_name": "x", "system_prompt": "y"}]}
        with self.assertRaises(ValueError, msg="name"):
            parse_agent_configs(config, ["a"], "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)