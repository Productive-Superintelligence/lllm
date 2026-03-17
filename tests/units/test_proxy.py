"""
Tests for lllm/proxies/base.py
Covers: BaseProxy, ProxyManager, ProxyRegistrator, register_proxy,
        endpoint decorator, endpoint_directory, api_directory, auto_test,
        retrieve_api_docs, dispatch.

Uses the real proxy classes from tests/test_cases/packages/pkg_delta/proxies/
as well as inline proxy definitions for isolated unit tests.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Make the pkg_delta proxy importable for integration tests
_PKG_DELTA_PROXIES = Path(__file__).parent.parent / "test_cases" / "packages" / "pkg_delta" / "proxies"


def _load_search_proxy():
    """Dynamically load SearchProxy from the real test_cases file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search_proxy",
        str(_PKG_DELTA_PROXIES / "search_proxy.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SearchProxy, mod.AnalyticsProxy


# ===========================================================================
# Inline proxy fixtures
# ===========================================================================


def _make_simple_proxy():
    """Define and return a minimal proxy class."""
    from lllm.proxies.base import BaseProxy

    class SimpleProxy(BaseProxy):
        _proxy_path = "simple"
        _proxy_name = "Simple Proxy"
        _proxy_description = "A simple test proxy"

        @BaseProxy.endpoint(
            category="data",
            endpoint="/items",
            description="List items",
            params={"limit": (int, 10)},
            response=[{"id": "str"}],
            method="GET",
        )
        def list_items(self, limit: int = 10):
            """List available items."""
            return [{"id": f"item_{i}"} for i in range(limit)]

        @BaseProxy.endpoint(
            category="data",
            endpoint="/items/{id}",
            description="Get a specific item",
            params={"item_id": (str, "item_0")},
            response={"id": "str", "data": "any"},
            method="GET",
        )
        def get_item(self, item_id: str):
            """Get a specific item by ID."""
            return {"id": item_id, "data": f"content of {item_id}"}

        def _private_method(self):
            """This should NOT appear in endpoint_directory."""
            return "private"

    return SimpleProxy


def _make_proxy_with_postcall():
    from lllm.proxies.base import BaseProxy

    class PostcallProxy(BaseProxy):
        _proxy_path = "postcall"
        _proxy_name = "Postcall Proxy"
        _proxy_description = "Proxy to test postcall decorator"

        @BaseProxy.endpoint(
            category="ops",
            endpoint="/op",
            description="An operation",
            params={},
            response={"ok": "bool"},
        )
        def do_op(self):
            return {"ok": True}

        @BaseProxy.postcall
        def cleanup(self):
            """Marked as postcall."""
            return "cleaned"

    return PostcallProxy


# ===========================================================================
# BaseProxy unit tests
# ===========================================================================


class TestBaseProxyConstruction(unittest.TestCase):

    def setUp(self):
        self.ProxyCls = _make_simple_proxy()

    def test_basic_construction(self):
        p = self.ProxyCls()
        self.assertEqual(p.activate_proxies, [])
        self.assertIsNone(p.cutoff_date)
        self.assertFalse(p.deploy_mode)
        self.assertTrue(p.use_cache)

    def test_construction_with_args(self):
        import datetime as dt
        p = self.ProxyCls(
            activate_proxies=["simple", "other"],
            deploy_mode=True,
            use_cache=False,
        )
        self.assertEqual(p.activate_proxies, ["simple", "other"])
        self.assertTrue(p.deploy_mode)
        self.assertFalse(p.use_cache)

    def test_activate_proxies_is_copy(self):
        original = ["simple"]
        p = self.ProxyCls(activate_proxies=original)
        original.append("injected")
        self.assertNotIn("injected", p.activate_proxies)

    def test_cutoff_date_str_parsed(self):
        import datetime as dt
        p = self.ProxyCls(cutoff_date="2024-01-15")
        self.assertIsInstance(p.cutoff_date, dt.datetime)

    def test_cutoff_date_invalid_str_becomes_none(self):
        p = self.ProxyCls(cutoff_date="not-a-date")
        self.assertIsNone(p.cutoff_date)


class TestEndpointDecorator(unittest.TestCase):

    def setUp(self):
        self.ProxyCls = _make_simple_proxy()
        self.proxy = self.ProxyCls()

    def test_endpoint_info_attached(self):
        method = self.proxy.list_items
        info = getattr(method, "endpoint_info", None)
        self.assertIsNotNone(info)
        self.assertEqual(info["category"], "data")
        self.assertEqual(info["endpoint"], "/items")
        self.assertEqual(info["method"], "GET")

    def test_endpoint_callable_normally(self):
        result = self.proxy.list_items(limit=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["id"], "item_0")

    def test_get_item_callable(self):
        result = self.proxy.get_item("item_5")
        self.assertEqual(result["id"], "item_5")

    def test_postcall_attribute(self):
        ProxyCls = _make_proxy_with_postcall()
        p = ProxyCls()
        self.assertTrue(getattr(p.cleanup, "is_postcall", False))


class TestEndpointDirectory(unittest.TestCase):

    def setUp(self):
        self.ProxyCls = _make_simple_proxy()
        self.proxy = self.ProxyCls()

    def test_returns_list(self):
        directory = self.proxy.endpoint_directory()
        self.assertIsInstance(directory, list)

    def test_only_endpoint_decorated_methods(self):
        directory = self.proxy.endpoint_directory()
        names = [e["callable"] for e in directory]
        self.assertIn("list_items", names)
        self.assertIn("get_item", names)
        self.assertNotIn("_private_method", names)

    def test_entries_have_required_keys(self):
        directory = self.proxy.endpoint_directory()
        for entry in directory:
            self.assertIn("category", entry)
            self.assertIn("endpoint", entry)
            self.assertIn("description", entry)
            self.assertIn("callable", entry)
            self.assertIn("docstring", entry)

    def test_sorted_by_category_then_endpoint(self):
        directory = self.proxy.endpoint_directory()
        for i in range(len(directory) - 1):
            a = (directory[i].get("category") or "", directory[i].get("endpoint") or "")
            b = (directory[i+1].get("category") or "", directory[i+1].get("endpoint") or "")
            self.assertLessEqual(a, b)

    def test_name_defaults_to_callable_name(self):
        directory = self.proxy.endpoint_directory()
        for entry in directory:
            self.assertIsNotNone(entry.get("name") or entry.get("callable"))


class TestApiDirectory(unittest.TestCase):

    def setUp(self):
        self.ProxyCls = _make_simple_proxy()
        self.proxy = self.ProxyCls()

    def test_api_directory_structure(self):
        meta = self.proxy.api_directory()
        self.assertIn("id", meta)
        self.assertIn("display_name", meta)
        self.assertIn("description", meta)
        self.assertIn("endpoints", meta)

    def test_id_from_proxy_path(self):
        meta = self.proxy.api_directory()
        self.assertEqual(meta["id"], "simple")

    def test_display_name_from_proxy_name(self):
        meta = self.proxy.api_directory()
        self.assertEqual(meta["display_name"], "Simple Proxy")

    def test_description_from_proxy_description(self):
        meta = self.proxy.api_directory()
        self.assertEqual(meta["description"], "A simple test proxy")


class TestAutoTest(unittest.TestCase):

    def setUp(self):
        self.ProxyCls = _make_simple_proxy()
        self.proxy = self.ProxyCls()

    def test_returns_dict(self):
        results = self.proxy.auto_test()
        self.assertIsInstance(results, dict)

    def test_status_ok_for_valid_endpoints(self):
        results = self.proxy.auto_test()
        for name, result in results.items():
            self.assertIn("status", result)
            self.assertIn("issues", result)
            self.assertIn(result["status"], ("ok", "warning"))

    def test_missing_params_detected(self):
        from lllm.proxies.base import BaseProxy

        class BrokenProxy(BaseProxy):
            @BaseProxy.endpoint(
                category="x",
                endpoint="/x",
                description="Broken",
                params=None,  # invalid!
                response=None,  # also invalid
            )
            def broken(self):
                return {}

        p = BrokenProxy()
        results = p.auto_test()
        self.assertIn("broken", results)
        self.assertEqual(results["broken"]["status"], "warning")
        self.assertIn("params", results["broken"]["issues"])
        self.assertIn("response", results["broken"]["issues"])


# ===========================================================================
# ProxyManager
# ===========================================================================


class TestProxyManager(unittest.TestCase):

    def _make_manager_with_proxy(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        ProxyCls = _make_simple_proxy()
        rt.register_proxy("simple", ProxyCls, overwrite=True)
        return ProxyManager(runtime=rt), rt

    def test_empty_manager(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        mgr = ProxyManager(runtime=rt)
        self.assertEqual(mgr.available(), [])

    def test_loads_registered_proxies(self):
        mgr, rt = self._make_manager_with_proxy()
        self.assertIn("simple", mgr.available())

    def test_available_sorted(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        ProxyCls = _make_simple_proxy()

        class OtherProxy(ProxyCls):
            _proxy_path = "zzz_other"

        rt.register_proxy("simple", ProxyCls, overwrite=True)
        rt.register_proxy("zzz_other", OtherProxy, overwrite=True)
        mgr = ProxyManager(runtime=rt)
        avail = mgr.available()
        self.assertEqual(avail, sorted(avail))

    def test_api_catalog_structure(self):
        mgr, _ = self._make_manager_with_proxy()
        catalog = mgr.api_catalog()
        self.assertIn("simple", catalog)
        self.assertIn("endpoints", catalog["simple"])

    def test_get_api_directory(self):
        mgr, _ = self._make_manager_with_proxy()
        directory = mgr.get_api_directory("simple")
        self.assertIn("endpoints", directory)

    def test_get_api_directory_not_found_raises(self):
        mgr, _ = self._make_manager_with_proxy()
        with self.assertRaises(KeyError):
            mgr.get_api_directory("nonexistent")

    def test_call_dispatch_dot_notation(self):
        mgr, _ = self._make_manager_with_proxy()
        result = mgr("simple.list_items", limit=2)
        self.assertEqual(len(result), 2)

    def test_call_dispatch_slash_notation(self):
        mgr, _ = self._make_manager_with_proxy()
        result = mgr("simple/list_items", limit=3)
        self.assertEqual(len(result), 3)

    def test_call_dispatch_nonexistent_proxy_raises(self):
        mgr, _ = self._make_manager_with_proxy()
        with self.assertRaises(KeyError):
            mgr("nonexistent.method")

    def test_call_dispatch_nonexistent_method_raises(self):
        mgr, _ = self._make_manager_with_proxy()
        with self.assertRaises(AttributeError):
            mgr("simple.nonexistent_method")

    def test_resolve_invalid_format_raises(self):
        mgr, _ = self._make_manager_with_proxy()
        with self.assertRaises(ValueError):
            mgr("no_separator")

    def test_retrieve_api_docs_all(self):
        mgr, _ = self._make_manager_with_proxy()
        docs = mgr.retrieve_api_docs()
        self.assertIsInstance(docs, str)
        self.assertIn("Simple Proxy", docs)
        self.assertIn("/items", docs)

    def test_retrieve_api_docs_specific(self):
        mgr, _ = self._make_manager_with_proxy()
        docs = mgr.retrieve_api_docs("simple")
        self.assertIn("Simple Proxy", docs)

    def test_retrieve_api_docs_not_found_raises(self):
        mgr, _ = self._make_manager_with_proxy()
        with self.assertRaises(KeyError):
            mgr.retrieve_api_docs("nonexistent")

    def test_register_new_proxy(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        mgr = ProxyManager(runtime=rt)
        ProxyCls = _make_simple_proxy()
        mgr.register("simple", ProxyCls)
        self.assertIn("simple", mgr.available())

    def test_activate_proxies_filter(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime

        class ProxyA(_make_simple_proxy()):
            _proxy_path = "proxy_a"

        class ProxyB(_make_simple_proxy()):
            _proxy_path = "proxy_b"

        rt = Runtime()
        rt.register_proxy("proxy_a", ProxyA, overwrite=True)
        rt.register_proxy("proxy_b", ProxyB, overwrite=True)

        # Only activate proxy_a
        mgr = ProxyManager(activate_proxies=["proxy_a"], runtime=rt)
        self.assertIn("proxy_a", mgr.available())
        self.assertNotIn("proxy_b", mgr.available())


# ===========================================================================
# ProxyRegistrator decorator
# ===========================================================================


class TestProxyRegistrator(unittest.TestCase):

    def test_registrator_sets_attrs(self):
        from lllm.proxies.base import ProxyRegistrator, BaseProxy
        from lllm.core.runtime import Runtime

        rt = Runtime()

        @ProxyRegistrator(path="my_proxy", name="My Proxy", description="Test proxy", runtime=rt)
        class MyProxy(BaseProxy):
            pass

        self.assertEqual(MyProxy._proxy_path, "my_proxy")
        self.assertEqual(MyProxy._proxy_name, "My Proxy")
        self.assertEqual(MyProxy._proxy_description, "Test proxy")

    def test_registrator_registers_to_runtime(self):
        from lllm.proxies.base import ProxyRegistrator, BaseProxy
        from lllm.core.runtime import Runtime

        rt = Runtime()

        @ProxyRegistrator(path="reg_test", name="Reg Test", description="desc", runtime=rt)
        class RegProxy(BaseProxy):
            pass

        self.assertTrue(rt.has("reg_test"))

    def test_register_proxy_function(self):
        from lllm.proxies.base import register_proxy, BaseProxy
        from lllm.core.runtime import get_default_runtime

        class FuncProxy(BaseProxy):
            pass

        # Should not raise; registers to default runtime
        try:
            register_proxy("func_proxy_test", FuncProxy, overwrite=False)
        except ValueError:
            # Already registered from a previous test run — that's fine
            pass


# ===========================================================================
# Real SearchProxy from test_cases
# ===========================================================================


class TestRealSearchProxy(unittest.TestCase):

    def setUp(self):
        SearchProxy, AnalyticsProxy = _load_search_proxy()
        self.SearchProxy = SearchProxy
        self.AnalyticsProxy = AnalyticsProxy
        self.proxy = SearchProxy()

    def test_search_returns_results(self):
        results = self.proxy.search("machine learning", limit=3)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertIn("id", r)
            self.assertIn("content", r)

    def test_index_returns_success(self):
        result = self.proxy.index("Test document content", doc_id="test_001")
        self.assertTrue(result["success"])
        self.assertEqual(result["indexed_id"], "test_001")

    def test_index_auto_id(self):
        result = self.proxy.index("Auto ID document")
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["indexed_id"])

    def test_get_stats(self):
        result = self.proxy.get_stats()
        self.assertIn("total_docs", result)
        self.assertIn("index_size_mb", result)

    def test_suggest_returns_list(self):
        result = self.proxy.suggest("que", max_suggestions=3)
        self.assertIsInstance(result, list)

    def test_endpoint_directory_has_all_endpoints(self):
        directory = self.proxy.endpoint_directory()
        endpoints = [e["callable"] for e in directory]
        self.assertIn("search", endpoints)
        self.assertIn("index", endpoints)
        self.assertIn("get_stats", endpoints)
        self.assertIn("suggest", endpoints)

    def test_analytics_proxy_track(self):
        ap = self.AnalyticsProxy()
        result = ap.track("click", properties={"button": "submit"})
        self.assertTrue(result["tracked"])

    def test_analytics_proxy_query_events(self):
        ap = self.AnalyticsProxy()
        result = ap.query_events("click")
        self.assertIn("count", result)

    def test_proxy_manager_with_real_proxies(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        rt.register_proxy("search", self.SearchProxy, overwrite=True)
        rt.register_proxy("analytics", self.AnalyticsProxy, overwrite=True)
        mgr = ProxyManager(runtime=rt)
        self.assertIn("search", mgr.available())
        self.assertIn("analytics", mgr.available())

        # Test dispatch
        results = mgr("search.search", query="test", limit=2)
        self.assertEqual(len(results), 2)

    def test_api_catalog_includes_real_proxies(self):
        from lllm.proxies.base import ProxyManager
        from lllm.core.runtime import Runtime
        rt = Runtime()
        rt.register_proxy("search", self.SearchProxy, overwrite=True)
        mgr = ProxyManager(runtime=rt)
        catalog = mgr.api_catalog()
        self.assertIn("search", catalog)
        self.assertEqual(catalog["search"]["display_name"], "Search Proxy")

    def test_auto_test_passes_for_real_proxy(self):
        results = self.proxy.auto_test()
        # All real endpoints have valid params/response
        for name, result in results.items():
            self.assertEqual(result["status"], "ok", f"Endpoint '{name}' has issues: {result['issues']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
