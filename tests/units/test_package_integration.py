"""
Integration tests using the REAL package structures in tests/test_cases/packages/.

Each test class loads one or more real packages from disk and verifies:
  - Resource registration and key naming
  - Config lazy loading and inheritance
  - Dependency resolution (including aliased deps)
  - Custom sections (assets, data)
  - Under-prefix path handling
  - Cycle detection (no infinite loop)
  - Prompt rendering and parsing with real content
  - Corner cases: empty package, hidden files, binary assets
"""
import unittest
from pathlib import Path

CASES_DIR = Path(__file__).parent.parent / "test_cases" / "packages"


def _load(toml_name, runtime):
    from lllm.core.config import load_package
    load_package(str(CASES_DIR / toml_name / "lllm.toml"), runtime=runtime)


def _fresh_rt():
    from lllm.core.runtime import Runtime
    return Runtime()


# ===========================================================================
# pkg_alpha — comprehensive prompts + configs
# ===========================================================================


class TestPkgAlpha(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_alpha", self.rt)

    # -- Package registration --

    def test_package_registered(self):
        self.assertIn("pkg_alpha", self.rt.packages)

    def test_default_namespace(self):
        self.assertEqual(self.rt._default_namespace, "pkg_alpha")

    def test_package_version(self):
        self.assertEqual(self.rt.packages["pkg_alpha"].version, "1.0.0")

    # -- Chat prompts (top-level prompts/chat.py) --

    def test_greet_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/greet"))

    def test_farewell_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/farewell"))

    def test_ask_topic_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/ask_topic"))

    def test_multi_var_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/multi_var"))

    def test_literal_braces_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/literal_braces"))

    def test_pkg_colon_shorthand(self):
        p = self.rt.get_prompt("pkg_alpha:chat/greet")
        self.assertEqual(p.path, "greet")

    def test_bare_key_resolves_via_default_namespace(self):
        p = self.rt.get_prompt("chat/greet")
        self.assertEqual(p.path, "greet")

    # -- Analysis prompts (nested subfolder) --

    def test_nested_summary_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:analysis/analyzer/summary"))

    def test_nested_multi_tag_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:analysis/analyzer/multi_tag"))

    def test_nested_code_review_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:analysis/analyzer/code_review"))

    def test_nested_signal_only_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:analysis/analyzer/signal_only"))

    # -- Template prompts (another subfolder) --

    def test_templates_system_base_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:templates/base/system_base"))

    def test_templates_user_task_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:templates/base/user_task"))

    def test_templates_with_custom_handler_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:templates/base/with_custom_handler"))

    # -- Prompt rendering --

    def test_static_prompt_renders_unchanged(self):
        p = self.rt.get_prompt("chat/greet")
        result = p()
        self.assertEqual(result, "Hello! How can I help you today?")

    def test_single_var_renders(self):
        p = self.rt.get_prompt("chat/ask_topic")
        result = p(topic="quantum computing")
        self.assertIn("quantum computing", result)

    def test_multi_var_renders(self):
        p = self.rt.get_prompt("chat/multi_var")
        result = p(subject="AI", viewpoint="an economist", language="English")
        self.assertIn("AI", result)
        self.assertIn("economist", result)

    def test_missing_var_raises(self):
        p = self.rt.get_prompt("chat/multi_var")
        with self.assertRaises(ValueError):
            p(subject="AI")  # missing viewpoint and language

    def test_literal_braces_not_treated_as_var(self):
        p = self.rt.get_prompt("chat/literal_braces")
        result = p(value="42")
        self.assertIn("{double braces}", result)
        self.assertIn("42", result)

    # -- Prompt template_vars --

    def test_greet_has_no_template_vars(self):
        p = self.rt.get_prompt("chat/greet")
        self.assertEqual(p.template_vars, set())

    def test_ask_topic_has_topic_var(self):
        p = self.rt.get_prompt("chat/ask_topic")
        self.assertIn("topic", p.template_vars)

    def test_multi_var_has_three_vars(self):
        p = self.rt.get_prompt("chat/multi_var")
        self.assertEqual(p.template_vars, {"subject", "viewpoint", "language"})

    # -- Parser on analysis prompts --

    def test_summary_has_parser(self):
        p = self.rt.get_prompt("analysis/analyzer/summary")
        self.assertIsNotNone(p.parser)

    def test_summary_parser_extracts_tag(self):
        p = self.rt.get_prompt("analysis/analyzer/summary")
        parsed = p.parse("<summary>This is a summary.</summary><key_points>Point 1</key_points>")
        self.assertEqual(parsed["xml_tags"]["summary"], ["This is a summary."])

    def test_signal_only_parser(self):
        p = self.rt.get_prompt("analysis/analyzer/signal_only")
        parsed = p.parse("This claim is <VALID>.")
        self.assertTrue(parsed["signal_tags"]["VALID"])
        self.assertFalse(parsed["signal_tags"]["INVALID"])

    # -- Configs lazy loading --

    def test_default_config_registered_lazily(self):
        node = self.rt.get_node("pkg_alpha.configs:default")
        self.assertFalse(node.is_loaded)

    def test_default_config_loads_on_access(self):
        node = self.rt.get_node("pkg_alpha.configs:default")
        cfg = self.rt.get_config("default")
        self.assertTrue(node.is_loaded)
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-mini")

    def test_default_config_agent_configs(self):
        cfg = self.rt.get_config("default")
        self.assertEqual(len(cfg["agent_configs"]), 1)
        self.assertEqual(cfg["agent_configs"][0]["name"], "assistant")

    def test_production_config_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.configs:production"))

    def test_development_config_registered(self):
        self.assertTrue(self.rt.has("pkg_alpha.configs:development"))

    # -- Config inheritance (resolve_config) --

    def test_production_inherits_from_default(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("production", self.rt)
        # From production: gpt-4o overrides default's gpt-4o-mini
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o")
        # temperature from production
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.1)
        # max_tokens from production
        self.assertEqual(cfg["global"]["model_args"]["max_tokens"], 4000)

    def test_development_inherits_temperature(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("development", self.rt)
        # development has temperature 0.95, max_tokens 500
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.95)
        # model_name from default (not overridden in development)
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-mini")

    def test_production_has_two_agents(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("production", self.rt)
        # production defines 2 agents (list replacement, not merge)
        names = {a["name"] for a in cfg["agent_configs"]}
        self.assertIn("assistant", names)
        self.assertIn("reviewer", names)

    def test_base_key_removed_after_resolution(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("production", self.rt)
        self.assertNotIn("base", cfg)

    # -- parse_agent_configs with real config --

    def test_parse_agent_configs_from_production(self):
        from lllm.core.config import resolve_config, parse_agent_configs
        cfg = resolve_config("production", self.rt)
        specs = parse_agent_configs(cfg, ["assistant", "reviewer"], "test_tactic")
        self.assertEqual(set(specs.keys()), {"assistant", "reviewer"})
        self.assertEqual(specs["assistant"].model, "gpt-4o")
        self.assertAlmostEqual(specs["reviewer"].model_args["temperature"], 0.0)

    # -- Prompt metadata --

    def test_system_base_metadata(self):
        p = self.rt.get_prompt("templates/base/system_base")
        self.assertEqual(p.metadata["type"], "system")
        self.assertEqual(p.metadata["version"], "1.0")


# ===========================================================================
# pkg_beta — depends on pkg_alpha
# ===========================================================================


class TestPkgBeta(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_beta", self.rt)

    def test_both_packages_registered(self):
        self.assertIn("pkg_beta", self.rt.packages)
        self.assertIn("pkg_alpha", self.rt.packages)

    def test_default_namespace_is_beta(self):
        """Root package's namespace wins."""
        self.assertEqual(self.rt._default_namespace, "pkg_beta")

    # -- Beta's own prompts --

    def test_chain_of_thought_registered(self):
        self.assertTrue(self.rt.has("pkg_beta.prompts:reasoning/chain_of_thought"))

    def test_socratic_registered(self):
        self.assertTrue(self.rt.has("pkg_beta.prompts:reasoning/socratic"))

    def test_debate_registered(self):
        self.assertTrue(self.rt.has("pkg_beta.prompts:reasoning/debate"))

    def test_hypothetical_registered(self):
        self.assertTrue(self.rt.has("pkg_beta.prompts:reasoning/hypothetical"))

    # -- Alpha's prompts accessible via alpha namespace --

    def test_alpha_prompts_accessible(self):
        self.assertTrue(self.rt.has("pkg_alpha.prompts:chat/greet"))

    def test_alpha_prompt_via_pkg_colon(self):
        p = self.rt.get_prompt("pkg_alpha:chat/greet")
        self.assertEqual(p.path, "greet")

    # -- Beta's config --

    def test_beta_config_registered(self):
        self.assertTrue(self.rt.has("pkg_beta.configs:beta_default"))

    def test_beta_config_content(self):
        cfg = self.rt.get_config("beta_default")
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o")
        self.assertEqual(len(cfg["agent_configs"]), 2)

    # -- Parse beta agent configs --

    def test_parse_beta_agent_configs(self):
        from lllm.core.config import parse_agent_configs
        cfg = self.rt.get_config("beta_default")
        specs = parse_agent_configs(cfg, ["reasoner", "critic"], "beta_tactic")
        self.assertIn("reasoner", specs)
        self.assertIn("critic", specs)
        self.assertEqual(specs["reasoner"].model, "gpt-4o")

    # -- Chain-of-thought prompt rendering --

    def test_cot_prompt_renders(self):
        p = self.rt.get_prompt("pkg_beta:reasoning/chain_of_thought")
        rendered = p(problem="What is the meaning of life?")
        self.assertIn("What is the meaning of life?", rendered)

    def test_debate_parser(self):
        p = self.rt.get_prompt("pkg_beta:reasoning/debate")
        self.assertIsNotNone(p.parser)
        parsed = p.parse("<pro>Better quality</pro><con>Higher cost</con>")
        self.assertEqual(parsed["xml_tags"]["pro"], ["Better quality"])
        self.assertEqual(parsed["xml_tags"]["con"], ["Higher cost"])

    def test_multiple_prompts_across_packages(self):
        """Both alpha and beta prompts are accessible in the same runtime."""
        alpha_p = self.rt.get_prompt("pkg_alpha:chat/greet")
        beta_p = self.rt.get_prompt("pkg_beta:reasoning/socratic")
        self.assertIsNot(alpha_p, beta_p)
        self.assertEqual(alpha_p.path, "greet")
        self.assertEqual(beta_p.path, "socratic")


# ===========================================================================
# pkg_gamma — custom sections (assets, data)
# ===========================================================================


class TestPkgGamma(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_gamma", self.rt)

    def test_package_registered(self):
        self.assertIn("pkg_gamma", self.rt.packages)

    # -- Assets section --

    def test_schema_json_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.assets:schema.json"))

    def test_schema_json_lazy_before_access(self):
        node = self.rt.get_node("pkg_gamma.assets:schema.json")
        self.assertFalse(node.is_loaded)

    def test_schema_json_loaded_on_access(self):
        data = self.rt.get("pkg_gamma.assets:schema.json")
        self.assertIsInstance(data, dict)
        self.assertEqual(data["title"], "AnalysisResult")
        self.assertIn("properties", data)

    def test_schema_json_node_loaded_after_access(self):
        self.rt.get("pkg_gamma.assets:schema.json")
        node = self.rt.get_node("pkg_gamma.assets:schema.json")
        self.assertTrue(node.is_loaded)

    def test_png_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.assets:logo.png"))

    def test_png_returns_bytes(self):
        data = self.rt.get("pkg_gamma.assets:logo.png")
        self.assertIsInstance(data, bytes)
        self.assertTrue(data.startswith(b'\x89PNG'))

    def test_nested_asset_yaml_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.assets:models/config.yaml"))

    def test_nested_asset_yaml_loaded(self):
        data = self.rt.get("pkg_gamma.assets:models/config.yaml")
        # YAML file should load as dict or bytes depending on implementation
        self.assertIsNotNone(data)

    def test_hidden_file_not_registered(self):
        self.assertFalse(self.rt.has("pkg_gamma.assets:.hidden_file"))

    def test_pycache_not_registered(self):
        # __pycache__ contents should be skipped
        # Check that no cache.pyc key exists
        keys = self.rt.keys()
        self.assertFalse(any("__pycache__" in k for k in keys))

    # -- Data section --

    def test_sample_json_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.data:sample.json"))

    def test_sample_json_content(self):
        data = self.rt.get("pkg_gamma.data:sample.json")
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]["id"], "s001")

    def test_nested_data_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.data:training/examples.json"))

    def test_nested_data_content(self):
        data = self.rt.get("pkg_gamma.data:training/examples.json")
        self.assertIsInstance(data, dict)
        self.assertEqual(data["version"], "1.0")
        self.assertEqual(data["total"], 3)

    # -- Prompts in gamma --

    def test_analyst_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.prompts:system/analyst"))

    def test_visualizer_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_gamma.prompts:system/visualizer"))

    # -- file_path metadata on assets --

    def test_asset_has_file_path_metadata(self):
        self.rt.get("pkg_gamma.assets:logo.png")  # trigger load
        node = self.rt.get_node("pkg_gamma.assets:logo.png")
        self.assertIn("file_path", node.metadata)


# ===========================================================================
# pkg_empty — no resources
# ===========================================================================


class TestPkgEmpty(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_empty", self.rt)

    def test_package_registered(self):
        self.assertIn("pkg_empty", self.rt.packages)

    def test_no_prompts(self):
        self.assertEqual(len(self.rt.keys("prompt")), 0)

    def test_no_configs(self):
        self.assertEqual(len(self.rt.keys("config")), 0)

    def test_has_no_resources(self):
        # Only the package itself is tracked, no resources
        self.assertEqual(len(self.rt._resources), 0)


# ===========================================================================
# pkg_prefixed — under-prefix path handling
# ===========================================================================


class TestPkgPrefixed(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_prefixed", self.rt)

    def test_package_registered(self):
        self.assertIn("pkg_prefixed", self.rt.packages)

    def test_prompts_under_tools_prefix(self):
        """All vendor_prompts are registered under 'tools/' prefix."""
        self.assertTrue(self.rt.has("pkg_prefixed.prompts:tools/tools/web_search"))

    def test_calculator_under_prefix(self):
        self.assertTrue(self.rt.has("pkg_prefixed.prompts:tools/tools/calculator"))

    def test_translator_under_prefix(self):
        self.assertTrue(self.rt.has("pkg_prefixed.prompts:tools/tools/translator"))

    def test_web_search_renders(self):
        p = self.rt.get_prompt("pkg_prefixed:tools/tools/web_search")
        result = p(query="latest AI news", n="5")
        self.assertIn("latest AI news", result)

    def test_no_prompts_without_prefix(self):
        """Prompts are NOT accessible without the 'tools' prefix."""
        self.assertFalse(self.rt.has("pkg_prefixed.prompts:vendor_prompts/tools/web_search"))


# ===========================================================================
# pkg_cycle_a + pkg_cycle_b — cycle detection
# ===========================================================================


class TestCycleDetection(unittest.TestCase):

    def test_cycle_does_not_hang(self):
        """Loading a cyclically-dependent package should terminate."""
        rt = _fresh_rt()
        _load("pkg_cycle_a", rt)  # loads B, which tries to load A again
        # If we reach here, no infinite loop occurred
        self.assertIn("pkg_cycle_a", rt.packages)
        self.assertIn("pkg_cycle_b", rt.packages)

    def test_both_packages_registered_once(self):
        rt = _fresh_rt()
        _load("pkg_cycle_a", rt)
        # Each package should only appear once
        self.assertEqual(len([k for k in rt.packages if k == "pkg_cycle_a"]), 1)
        self.assertEqual(len([k for k in rt.packages if k == "pkg_cycle_b"]), 1)

    def test_load_from_b_side(self):
        """Loading from B's side should also terminate and register both."""
        rt = _fresh_rt()
        _load("pkg_cycle_b", rt)
        self.assertIn("pkg_cycle_a", rt.packages)
        self.assertIn("pkg_cycle_b", rt.packages)


# ===========================================================================
# pkg_delta — proxies
# ===========================================================================


class TestPkgDelta(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_delta", self.rt)

    def test_package_registered(self):
        self.assertIn("pkg_delta", self.rt.packages)

    def test_agent_prompts_registered(self):
        self.assertTrue(self.rt.has("pkg_delta.prompts:agent/search_agent"))
        self.assertTrue(self.rt.has("pkg_delta.prompts:agent/index_agent"))

    def test_search_agent_renders(self):
        p = self.rt.get_prompt("pkg_delta:agent/search_agent")
        result = p(query="machine learning trends")
        self.assertIn("machine learning trends", result)


# ===========================================================================
# load_resource convenience API with real packages
# ===========================================================================


class TestLoadResourceWithRealPackages(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_gamma", self.rt)

    def test_load_resource_full_url(self):
        from lllm.core.resource import load_resource
        data = load_resource("pkg_gamma.data:sample.json", runtime=self.rt)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)

    def test_load_resource_section_only(self):
        from lllm.core.resource import load_resource
        # Default namespace is pkg_gamma
        data = load_resource("data:sample.json", runtime=self.rt)
        self.assertIsInstance(data, list)

    def test_load_resource_missing_colon_raises(self):
        from lllm.core.resource import load_resource
        with self.assertRaises(ValueError):
            load_resource("no_colon_path", runtime=self.rt)

    def test_load_resource_nonexistent_raises(self):
        from lllm.core.resource import load_resource
        with self.assertRaises(KeyError):
            load_resource("pkg_gamma.data:nonexistent_resource.json", runtime=self.rt)


# ===========================================================================
# load_runtime with real packages
# ===========================================================================


class TestLoadRuntimeWithRealPackage(unittest.TestCase):

    def test_load_runtime_from_toml(self):
        from lllm.core.runtime import load_runtime
        toml = str(CASES_DIR / "pkg_alpha" / "lllm.toml")
        rt = load_runtime(toml_path=toml, name="test_alpha_runtime")
        self.assertTrue(rt._discovery_done)
        self.assertIn("pkg_alpha", rt.packages)
        self.assertTrue(rt.has("pkg_alpha.prompts:chat/greet"))

    def test_load_runtime_named_accessible(self):
        from lllm.core.runtime import load_runtime, get_runtime
        toml = str(CASES_DIR / "pkg_alpha" / "lllm.toml")
        load_runtime(toml_path=toml, name="test_named_runtime_alpha")
        rt = get_runtime("test_named_runtime_alpha")
        self.assertIn("pkg_alpha", rt.packages)

    def test_load_runtime_nonexistent_named_raises(self):
        from lllm.core.runtime import get_runtime
        with self.assertRaises(KeyError):
            get_runtime("definitely_nonexistent_runtime_xyz")


# ===========================================================================
# Cross-package: vendor_config with real packages
# ===========================================================================


class TestVendorConfigWithRealPackages(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_alpha", self.rt)

    def test_vendor_config_no_overrides(self):
        from lllm.core.config import vendor_config
        cfg = vendor_config("pkg_alpha:default", runtime=self.rt)
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-mini")

    def test_vendor_config_with_overrides(self):
        from lllm.core.config import vendor_config
        cfg = vendor_config("pkg_alpha:default", {
            "global": {"model_name": "gpt-4o-turbo"},
        }, runtime=self.rt)
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-turbo")
        # temperature from default preserved
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.7)

    def test_vendor_config_resolves_base_then_overrides(self):
        from lllm.core.config import vendor_config
        # production inherits from default, then we apply overrides on top
        cfg = vendor_config("pkg_alpha:production", {
            "global": {"model_args": {"temperature": 0.5}},
        }, runtime=self.rt)
        # Base chain: default -> production -> overrides
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o")     # from production
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.5)  # from override
        self.assertEqual(cfg["global"]["model_args"]["max_tokens"], 4000)  # from production


# ===========================================================================
# Multiple package loads into same runtime
# ===========================================================================


class TestMultiplePackagesInOneRuntime(unittest.TestCase):

    def test_alpha_and_gamma_coexist(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_gamma", rt)

        # Both packages registered
        self.assertIn("pkg_alpha", rt.packages)
        self.assertIn("pkg_gamma", rt.packages)

        # Resources from both accessible
        self.assertTrue(rt.has("pkg_alpha.prompts:chat/greet"))
        self.assertTrue(rt.has("pkg_gamma.data:sample.json"))

    def test_default_namespace_is_first_loaded(self):
        """First loaded package wins the default namespace."""
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_gamma", rt)
        # alpha was loaded first
        self.assertEqual(rt._default_namespace, "pkg_alpha")

    def test_resources_dont_collide(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_gamma", rt)

        alpha_prompt = rt.get_prompt("pkg_alpha:chat/greet")
        gamma_prompt = rt.get_prompt("pkg_gamma:system/analyst")
        self.assertIsNot(alpha_prompt, gamma_prompt)


# ===========================================================================
# PackageInfo properties
# ===========================================================================


class TestPackageInfoFromRealPackages(unittest.TestCase):

    def test_alpha_package_info(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        info = rt.packages["pkg_alpha"]
        self.assertEqual(info.name, "pkg_alpha")
        self.assertEqual(info.version, "1.0.0")
        self.assertEqual(info.description, "Alpha test package with prompts, configs, and template vars")
        self.assertIsNone(info.alias)
        self.assertEqual(info.effective_name, "pkg_alpha")

    def test_gamma_package_info(self):
        rt = _fresh_rt()
        _load("pkg_gamma", rt)
        info = rt.packages["pkg_gamma"]
        self.assertEqual(info.name, "pkg_gamma")
        # version not set → empty string
        self.assertEqual(info.version, "")


# ===========================================================================
# Runtime.reset() with real package data
# ===========================================================================


class TestRuntimeResetWithPackages(unittest.TestCase):

    def test_reset_clears_all(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)

        self.assertTrue(rt.has("pkg_alpha.prompts:chat/greet"))
        rt.reset()

        self.assertFalse(rt.has("pkg_alpha.prompts:chat/greet"))
        self.assertEqual(len(rt.packages), 0)
        self.assertIsNone(rt._default_namespace)
        self.assertFalse(rt._discovery_done)

    def test_can_reload_after_reset(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        rt.reset()
        _load("pkg_alpha", rt)
        self.assertTrue(rt.has("pkg_alpha.prompts:chat/greet"))


# ===========================================================================
# Keys and type filtering
# ===========================================================================


class TestKeysFiltering(unittest.TestCase):

    def test_prompt_keys_only(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        prompt_keys = rt.keys("prompt")
        config_keys = rt.keys("config")
        for k in prompt_keys:
            self.assertNotIn(k, config_keys)

    def test_config_keys_only(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        config_keys = rt.keys("config")
        self.assertGreater(len(config_keys), 0)
        for k in config_keys:
            self.assertIn("configs:", k)

    def test_all_keys_includes_everything(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        all_keys = rt.keys()
        prompt_keys = rt.keys("prompt")
        config_keys = rt.keys("config")
        for k in prompt_keys + config_keys:
            self.assertIn(k, all_keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
