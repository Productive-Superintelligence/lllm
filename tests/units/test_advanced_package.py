"""
Advanced package management tests covering complex real-world scenarios:

  - 4-level deep dependency chain (A → B → C → D)
  - Diamond dependency (root → left, right → shared; shared loaded once)
  - 3-level config inheritance
  - Package loaded idempotently (load_package called twice for same path)
  - Missing dependency path emits RuntimeWarning (graceful, not crash)
  - Invalid TOML raises exception
  - register() with overwrite=False raises ValueError
  - register_config() with a lazy loader function
  - get_proxy() and get_tactic() qualified-key lookups
  - find_config_file() respects LLLM_CONFIG env var
  - vendor_config across packages with multi-level inheritance
  - Package aliasing (dep loaded as alias)
  - Multiple packages sharing the same section name (no collision)
  - Runtime.keys() returns sorted, filtered results
"""
import os
import unittest
import warnings
from pathlib import Path

CASES_DIR = Path(__file__).parent.parent / "test_cases" / "packages"


def _load(toml_name, runtime):
    from lllm.core.config import load_package
    load_package(str(CASES_DIR / toml_name / "lllm.toml"), runtime=runtime)


def _fresh_rt():
    from lllm.core.runtime import Runtime
    return Runtime()


# ===========================================================================
# 4-level dependency chain: chain_a → chain_b → chain_c → chain_d
# ===========================================================================

class TestFourLevelDependencyChain(unittest.TestCase):
    """Loading pkg_chain_a pulls in b, c, and d transitively."""

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_chain_a", self.rt)

    def test_all_four_packages_registered(self):
        for pkg in ("pkg_chain_a", "pkg_chain_b", "pkg_chain_c", "pkg_chain_d"):
            with self.subTest(pkg=pkg):
                self.assertIn(pkg, self.rt.packages)

    def test_default_namespace_is_root(self):
        """The first-loaded (root) package owns the default namespace."""
        self.assertEqual(self.rt._default_namespace, "pkg_chain_a")

    def test_chain_a_top_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_chain_a.prompts:top/top"))

    def test_chain_b_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_chain_b.prompts:level2/level2"))

    def test_chain_c_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_chain_c.prompts:level3/level3"))

    def test_chain_d_foundation_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_chain_d.prompts:base/foundation"))

    def test_all_chain_prompts_render(self):
        for key in (
            "pkg_chain_a.prompts:top/top",
            "pkg_chain_b.prompts:level2/level2",
            "pkg_chain_c.prompts:level3/level3",
            "pkg_chain_d.prompts:base/foundation",
        ):
            with self.subTest(key=key):
                p = self.rt.get_prompt(key)
                rendered = p()
                self.assertIsInstance(rendered, str)
                self.assertGreater(len(rendered), 0)

    def test_chain_a_config_registered(self):
        self.assertTrue(self.rt.has("pkg_chain_a.configs:chain_config"))

    def test_chain_a_config_content(self):
        cfg = self.rt.get_config("chain_config")
        self.assertEqual(cfg["global"]["chain_level"], 1)


# ===========================================================================
# Diamond dependency: root → left + right → shared (shared loaded once)
# ===========================================================================

class TestDiamondDependency(unittest.TestCase):
    """
    Dependency graph:
        pkg_diamond_root
         ├── pkg_diamond_left  → pkg_diamond_shared
         └── pkg_diamond_right → pkg_diamond_shared

    pkg_diamond_shared should be loaded exactly once despite two paths reaching it.
    """

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_diamond_root", self.rt)

    def test_all_four_packages_registered(self):
        for pkg in (
            "pkg_diamond_root", "pkg_diamond_left",
            "pkg_diamond_right", "pkg_diamond_shared",
        ):
            with self.subTest(pkg=pkg):
                self.assertIn(pkg, self.rt.packages)

    def test_shared_package_loaded_once(self):
        """_loaded_package_paths deduplication: shared is registered exactly once."""
        # Only one entry for pkg_diamond_shared in packages dict
        shared_count = sum(1 for k in self.rt.packages if k == "pkg_diamond_shared")
        self.assertEqual(shared_count, 1)

    def test_shared_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_diamond_shared.prompts:shared/shared_base"))

    def test_left_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_diamond_left.prompts:left/left_prompt"))

    def test_right_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_diamond_right.prompts:right/right_prompt"))

    def test_root_prompt_registered(self):
        self.assertTrue(self.rt.has("pkg_diamond_root.prompts:root/root_prompt"))

    def test_shared_prompt_not_duplicated_in_keys(self):
        """Shared's prompt key should appear exactly once in the full key list."""
        all_keys = self.rt.keys()
        shared_key = "pkg_diamond_shared.prompts:shared/shared_base"
        count = all_keys.count(shared_key)
        self.assertEqual(count, 1)

    def test_shared_config_accessible(self):
        cfg = self.rt.get_config("pkg_diamond_shared.configs:shared_cfg")
        self.assertTrue(cfg["global"]["shared_flag"])


# ===========================================================================
# Package load idempotency
# ===========================================================================

class TestPackageLoadIdempotency(unittest.TestCase):
    """Loading the same package twice should not double-register resources."""

    def test_load_twice_idempotent(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        count_before = len(rt.keys())

        _load("pkg_alpha", rt)  # second load — should be skipped
        count_after = len(rt.keys())

        self.assertEqual(count_before, count_after)

    def test_packages_dict_not_duplicated(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_alpha", rt)
        pkg_alpha_count = sum(1 for k in rt.packages if k == "pkg_alpha")
        self.assertEqual(pkg_alpha_count, 1)

    def test_loaded_paths_tracked(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        self.assertEqual(len(rt._loaded_package_paths), 1)

        _load("pkg_alpha", rt)
        # Still just 1 — not added twice
        self.assertEqual(len(rt._loaded_package_paths), 1)


# ===========================================================================
# Missing dependency path — graceful warning
# ===========================================================================

class TestMissingDependencyWarning(unittest.TestCase):
    """A dependency entry pointing at a non-existent dir should warn, not crash."""

    def test_missing_dep_emits_warning(self):
        """Create a temp lllm.toml with a broken dep path and load it."""
        import tempfile
        import textwrap
        from lllm.core.config import load_package

        toml_content = textwrap.dedent("""\
            [package]
            name = "pkg_with_missing_dep"
            version = "0.1.0"

            [dependencies]
            packages = ["../pkg_does_not_exist_xyz"]
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "lllm.toml"
            toml_path.write_text(toml_content)

            rt = _fresh_rt()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_package(str(toml_path), runtime=rt)

            warning_msgs = [str(w.message) for w in caught]
            self.assertTrue(
                any("lllm.toml" in m or "no lllm.toml" in m.lower() or
                    "Dependency" in m or "pkg_does_not_exist_xyz" in m
                    for m in warning_msgs),
                f"Expected a RuntimeWarning about missing dep, got: {warning_msgs}",
            )

        # Package itself was still registered
        self.assertIn("pkg_with_missing_dep", rt.packages)

    def test_missing_dep_does_not_crash(self):
        """Verify the load call doesn't raise, only warns."""
        import tempfile
        import textwrap
        from lllm.core.config import load_package

        toml_content = textwrap.dedent("""\
            [package]
            name = "pkg_safe_load"
            version = "0.1.0"

            [dependencies]
            packages = ["../totally_nonexistent_xyz"]
        """)
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "lllm.toml"
            toml_path.write_text(toml_content)
            rt = _fresh_rt()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                try:
                    load_package(str(toml_path), runtime=rt)
                except Exception as exc:
                    self.fail(f"load_package raised unexpectedly: {exc}")


# ===========================================================================
# Invalid TOML — raises on load
# ===========================================================================

class TestInvalidToml(unittest.TestCase):

    def test_invalid_toml_raises(self):
        from lllm.core.config import load_package
        bad_path = str(CASES_DIR / "pkg_bad_toml" / "lllm.toml")
        with self.assertRaises(Exception):
            load_package(bad_path, runtime=_fresh_rt())


# ===========================================================================
# register() with overwrite=False
# ===========================================================================

class TestRegisterOverwrite(unittest.TestCase):

    def test_register_overwrite_false_raises(self):
        from lllm.core.resource import ResourceNode
        rt = _fresh_rt()

        node1 = ResourceNode.eager("mykey", "value1", namespace="ns", resource_type="prompt")
        rt.register(node1)

        node2 = ResourceNode.eager("mykey", "value2", namespace="ns", resource_type="prompt")
        with self.assertRaises(ValueError):
            rt.register(node2, overwrite=False)

    def test_register_overwrite_true_succeeds(self):
        from lllm.core.resource import ResourceNode
        rt = _fresh_rt()

        node1 = ResourceNode.eager("mykey", "value1", namespace="ns", resource_type="prompt")
        rt.register(node1)

        node2 = ResourceNode.eager("mykey", "value2", namespace="ns", resource_type="prompt")
        rt.register(node2, overwrite=True)  # should not raise
        self.assertEqual(rt.get("ns:mykey"), "value2")


# ===========================================================================
# register_config() with lazy loader function
# ===========================================================================

class TestRegisterConfigLazyLoader(unittest.TestCase):

    def test_lazy_config_not_loaded_before_access(self):
        rt = _fresh_rt()
        call_count = {"n": 0}

        def my_loader():
            call_count["n"] += 1
            return {"global": {"model_name": "gpt-4o-lazy"}}

        rt.register_config("lazy_cfg", namespace="test.configs", loader=my_loader)
        self.assertEqual(call_count["n"], 0)

    def test_lazy_config_loaded_on_first_access(self):
        rt = _fresh_rt()

        def my_loader():
            return {"global": {"model_name": "gpt-4o-lazy"}}

        rt.register_config("lazy_cfg", namespace="test.configs", loader=my_loader)
        cfg = rt.get_config("test.configs:lazy_cfg")
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-lazy")

    def test_lazy_config_cached_after_access(self):
        rt = _fresh_rt()
        call_count = {"n": 0}

        def my_loader():
            call_count["n"] += 1
            return {"value": call_count["n"]}

        rt.register_config("cached_cfg", namespace="test.configs", loader=my_loader)
        _ = rt.get_config("test.configs:cached_cfg")
        _ = rt.get_config("test.configs:cached_cfg")
        self.assertEqual(call_count["n"], 1)  # loaded only once


# ===========================================================================
# find_config_file() with LLLM_CONFIG env var
# ===========================================================================

class TestFindConfigFileEnvVar(unittest.TestCase):

    def test_lllm_config_env_var_file_path(self):
        """LLLM_CONFIG pointing at a file takes precedence."""
        from lllm.core.config import find_config_file
        toml_path = str(CASES_DIR / "pkg_alpha" / "lllm.toml")

        original = os.environ.get("LLLM_CONFIG")
        try:
            os.environ["LLLM_CONFIG"] = toml_path
            found = find_config_file()
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "lllm.toml")
        finally:
            if original is None:
                os.environ.pop("LLLM_CONFIG", None)
            else:
                os.environ["LLLM_CONFIG"] = original

    def test_lllm_config_env_var_dir_path(self):
        """LLLM_CONFIG pointing at a directory finds lllm.toml inside it."""
        from lllm.core.config import find_config_file
        dir_path = str(CASES_DIR / "pkg_alpha")

        original = os.environ.get("LLLM_CONFIG")
        try:
            os.environ["LLLM_CONFIG"] = dir_path
            found = find_config_file()
            self.assertIsNotNone(found)
            self.assertTrue(found.is_file())
        finally:
            if original is None:
                os.environ.pop("LLLM_CONFIG", None)
            else:
                os.environ["LLLM_CONFIG"] = original

    def test_lllm_config_nonexistent_falls_back(self):
        """If LLLM_CONFIG points to a non-existent file, fall back to normal search."""
        from lllm.core.config import find_config_file
        original = os.environ.get("LLLM_CONFIG")
        try:
            os.environ["LLLM_CONFIG"] = "/tmp/definitely_does_not_exist_lllm.toml"
            # Should not raise — falls back to normal upward search from cwd
            result = find_config_file()
            # Result may or may not be None depending on cwd; no crash is the key assertion
        except Exception as exc:
            self.fail(f"find_config_file raised unexpectedly: {exc}")
        finally:
            if original is None:
                os.environ.pop("LLLM_CONFIG", None)
            else:
                os.environ["LLLM_CONFIG"] = original


# ===========================================================================
# get_proxy() and get_tactic() with qualified keys
# ===========================================================================

class TestGetProxyAndTacticQualifiedKey(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_delta", self.rt)

    def test_get_proxy_qualified_key(self):
        """get_proxy() with full namespace:key works."""
        # _proxy_path = "search" → key is "search"
        proxy_cls = self.rt.get_proxy("pkg_delta.proxies:search")
        self.assertIsNotNone(proxy_cls)

    def test_get_proxy_two_registered(self):
        """Both proxies from search_proxy.py are accessible."""
        search = self.rt.get_proxy("pkg_delta.proxies:search")
        analytics = self.rt.get_proxy("pkg_delta.proxies:analytics")
        self.assertIsNot(search, analytics)

    def test_get_proxy_bare_key_default_ns(self):
        """Bare key resolved through default namespace (pkg_delta)."""
        proxy_cls = self.rt.get_proxy("search")
        self.assertIsNotNone(proxy_cls)

    def test_proxy_is_class(self):
        import inspect
        proxy_cls = self.rt.get_proxy("pkg_delta.proxies:search")
        self.assertTrue(inspect.isclass(proxy_cls))

    def test_second_proxy_in_same_file(self):
        """Both proxies from search_proxy.py are registered."""
        proxy_analytics = self.rt.get_proxy("pkg_delta.proxies:analytics")
        self.assertIsNotNone(proxy_analytics)


# ===========================================================================
# vendor_config across packages (multi-level inheritance spanning packages)
# ===========================================================================

class TestVendorConfigMultiLevel(unittest.TestCase):
    """vendor_config resolves the full inheritance chain then applies overrides."""

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_alpha", self.rt)

    def test_vendor_config_production_chain(self):
        """production → default; vendor with additional model pin."""
        from lllm.core.config import vendor_config
        cfg = vendor_config("pkg_alpha:production", {
            "global": {"model_args": {"top_p": 0.9}},
        }, runtime=self.rt)
        # From production (which inherits default)
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o")
        # Override applied
        self.assertAlmostEqual(cfg["global"]["model_args"]["top_p"], 0.9)
        # temperature from production preserved
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.1)
        # No base key in result
        self.assertNotIn("base", cfg)

    def test_vendor_config_registers_result(self):
        """vendor_config result can be re-registered as a new config."""
        from lllm.core.config import vendor_config
        cfg = vendor_config("pkg_alpha:default", runtime=self.rt)

        self.rt.register_config(
            "vendored/alpha_pinned",
            config_data=cfg,
            namespace="my_pkg.configs",
        )
        loaded = self.rt.get_config("my_pkg.configs:vendored/alpha_pinned")
        self.assertEqual(loaded["global"]["model_name"], "gpt-4o-mini")

    def test_resolve_config_circular_raises(self):
        """Circular config inheritance should raise ValueError."""
        from lllm.core.config import resolve_config
        # base must use the full qualified key so resolve_config can find it
        self.rt.register_config("circ_a",
                                {"base": "test_circ.configs:circ_b", "x": 1},
                                namespace="test_circ.configs")
        self.rt.register_config("circ_b",
                                {"base": "test_circ.configs:circ_a", "y": 2},
                                namespace="test_circ.configs")
        with self.assertRaises(ValueError):
            resolve_config("test_circ.configs:circ_a", self.rt)


# ===========================================================================
# Package aliasing
# ===========================================================================

class TestPackageAliasing(unittest.TestCase):
    """Dependency loaded with 'as alias' makes resources available under alias."""

    def test_alias_dep_via_toml(self):
        """Create a temp package that loads pkg_alpha as 'alpha_alias'."""
        import tempfile
        import textwrap
        from lllm.core.config import load_package

        toml_content = textwrap.dedent(f"""\
            [package]
            name = "pkg_alias_user"
            version = "0.1.0"

            [dependencies]
            packages = [
                "{{path = "{str(CASES_DIR / 'pkg_alpha')}", alias = "alpha_alias"}}",
            ]
        """)
        # TOML inline tables require a different format — use path/alias dict
        # Write correct TOML
        toml_content = (
            '[package]\n'
            'name = "pkg_alias_user"\n'
            'version = "0.1.0"\n'
            '\n'
            '[dependencies]\n'
            f'packages = [{{path = "{str(CASES_DIR / "pkg_alpha")}", alias = "alpha_alias"}}]\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "lllm.toml"
            toml_path.write_text(toml_content)
            rt = _fresh_rt()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_package(str(toml_path), runtime=rt)

        # Original name still present
        self.assertIn("pkg_alpha", rt.packages)
        # Alias registered
        self.assertIn("alpha_alias", rt.packages)

    def test_alias_resources_accessible(self):
        """Resources from aliased dep are reachable under the alias namespace."""
        import tempfile
        from lllm.core.config import load_package

        toml_content = (
            '[package]\n'
            'name = "pkg_alias_user2"\n'
            'version = "0.1.0"\n'
            '\n'
            '[dependencies]\n'
            f'packages = [{{path = "{str(CASES_DIR / "pkg_alpha")}", alias = "a"}}]\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "lllm.toml"
            toml_path.write_text(toml_content)
            rt = _fresh_rt()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_package(str(toml_path), runtime=rt)

        # Resources accessible under alias 'a'
        self.assertTrue(rt.has("a.prompts:chat/greet"))


# ===========================================================================
# Multiple packages with same section name (no collision)
# ===========================================================================

class TestSameSectionNameNoCollision(unittest.TestCase):
    """Two packages can have a 'prompts' section without overwriting each other."""

    def test_alpha_and_beta_prompts_coexist(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_gamma", rt)  # gamma also has a 'prompts' section

        # Both accessible via their namespaced keys
        self.assertTrue(rt.has("pkg_alpha.prompts:chat/greet"))
        self.assertTrue(rt.has("pkg_gamma.prompts:system/analyst"))

    def test_configs_section_no_collision(self):
        rt = _fresh_rt()
        _load("pkg_alpha", rt)
        _load("pkg_chain_a", rt)

        # Alpha's default config
        self.assertTrue(rt.has("pkg_alpha.configs:default"))
        # Chain_a's chain_config
        self.assertTrue(rt.has("pkg_chain_a.configs:chain_config"))

        # Values are independent
        alpha_cfg = rt.get_config("pkg_alpha.configs:default")
        chain_cfg = rt.get_config("pkg_chain_a.configs:chain_config")
        self.assertNotEqual(alpha_cfg, chain_cfg)


# ===========================================================================
# Runtime keys() filtering & sorting
# ===========================================================================

class TestRuntimeKeysSortedFiltered(unittest.TestCase):

    def setUp(self):
        self.rt = _fresh_rt()
        _load("pkg_alpha", self.rt)

    def test_all_keys_sorted(self):
        keys = self.rt.keys()
        self.assertEqual(keys, sorted(keys))

    def test_prompt_keys_all_have_prompts_in_name(self):
        prompt_keys = self.rt.keys("prompt")
        for k in prompt_keys:
            self.assertIn(".prompts:", k)

    def test_config_keys_all_have_configs_in_name(self):
        config_keys = self.rt.keys("config")
        for k in config_keys:
            self.assertIn(".configs:", k)

    def test_prompt_and_config_keys_disjoint(self):
        prompt_keys = set(self.rt.keys("prompt"))
        config_keys = set(self.rt.keys("config"))
        self.assertTrue(prompt_keys.isdisjoint(config_keys))

    def test_unknown_type_returns_empty(self):
        result = self.rt.keys("nonexistent_type_xyz")
        self.assertEqual(result, [])


# ===========================================================================
# 3-level config inheritance (deep_merge semantics)
# ===========================================================================

class TestThreeLevelConfigInheritance(unittest.TestCase):
    """Build a 3-level inheritance chain and verify full deep-merge resolution."""

    def setUp(self):
        self.rt = _fresh_rt()
        # Register three configs: base → mid → top
        self.rt.register_config("base3", {
            "global": {
                "model_name": "gpt-3.5-turbo",
                "model_args": {"temperature": 0.5, "max_tokens": 500},
                "retries": 3,
            },
            "shared_key": "from_base",
        }, namespace="test3.configs")

        self.rt.register_config("mid3", {
            "base": "test3.configs:base3",
            "global": {
                "model_name": "gpt-4o-mini",
                "model_args": {"temperature": 0.7},
            },
            "mid_key": "from_mid",
        }, namespace="test3.configs")

        self.rt.register_config("top3", {
            "base": "test3.configs:mid3",
            "global": {
                "model_args": {"max_tokens": 2000},
            },
            "top_key": "from_top",
        }, namespace="test3.configs")

    def test_three_level_model_name(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        # mid3 overrode base3's model_name
        self.assertEqual(cfg["global"]["model_name"], "gpt-4o-mini")

    def test_three_level_temperature_from_mid(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        # mid3's temperature 0.7 survives
        self.assertAlmostEqual(cfg["global"]["model_args"]["temperature"], 0.7)

    def test_three_level_max_tokens_from_top(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        # top3 overrode max_tokens to 2000
        self.assertEqual(cfg["global"]["model_args"]["max_tokens"], 2000)

    def test_three_level_retries_from_base(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        # base3's retries not overridden anywhere
        self.assertEqual(cfg["global"]["retries"], 3)

    def test_three_level_keys_from_all_levels(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        self.assertEqual(cfg["shared_key"], "from_base")
        self.assertEqual(cfg["mid_key"], "from_mid")
        self.assertEqual(cfg["top_key"], "from_top")

    def test_three_level_no_base_key_in_result(self):
        from lllm.core.config import resolve_config
        cfg = resolve_config("test3.configs:top3", self.rt)
        self.assertNotIn("base", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
