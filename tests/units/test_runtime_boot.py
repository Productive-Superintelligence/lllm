import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "tests" / "test_cases" / "packages"


class TestRuntimeIntrospection(unittest.TestCase):

    def test_iter_items_filters_by_resource_type(self):
        from lllm.core.resource import ResourceNode
        from lllm.core.runtime import Runtime

        rt = Runtime()
        rt.register(
            ResourceNode.eager(
                "chat/greet",
                object(),
                namespace="demo.prompts",
                resource_type="prompt",
            )
        )
        rt.register(
            ResourceNode.eager(
                "search",
                object(),
                namespace="demo.proxies",
                resource_type="proxy",
            )
        )

        items = list(rt.iter_items("proxy"))
        self.assertEqual([key for key, _ in items], ["demo.proxies:search"])

    def test_runtime_public_properties_start_empty(self):
        from lllm.core.runtime import Runtime

        rt = Runtime()
        self.assertIsNone(rt.default_namespace)
        self.assertFalse(rt.discovery_done)
        self.assertEqual(rt.loaded_package_paths, frozenset())


class TestLoadRuntimeStrictBoot(unittest.TestCase):

    def test_empty_runtime_when_implicit_discovery_disabled(self):
        from lllm.core.runtime import load_runtime

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = Path.cwd()
            try:
                os.chdir(tmp_dir)
                rt = load_runtime(
                    name="strict_boot_empty",
                    discover_cwd=False,
                    discover_shared_packages=False,
                )
            finally:
                os.chdir(cwd)

        self.assertEqual(rt.keys(), [])
        self.assertIsNone(rt.default_namespace)
        self.assertTrue(rt.discovery_done)

    def test_shared_package_autoload_can_be_disabled(self):
        from lllm.core.runtime import load_runtime

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shared_pkg_dir = tmp_path / "lllm_packages" / "pkg_alpha"
            shutil.copytree(CASES_DIR / "pkg_alpha", shared_pkg_dir)

            cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                strict_rt = load_runtime(
                    name="strict_no_shared",
                    discover_cwd=False,
                    discover_shared_packages=False,
                )
                shared_rt = load_runtime(
                    name="strict_with_shared",
                    discover_cwd=False,
                    discover_shared_packages=True,
                )
            finally:
                os.chdir(cwd)

        self.assertEqual(strict_rt.packages, {})
        self.assertIn("pkg_alpha", shared_rt.packages)
        self.assertTrue(shared_rt.has("pkg_alpha.prompts:chat/greet"))


class TestImportTimeControls(unittest.TestCase):

    def _run_python(self, code: str, *, env: dict[str, str] | None = None, cwd: Path) -> dict:
        subprocess_env = os.environ.copy()
        python_path = subprocess_env.get("PYTHONPATH")
        subprocess_env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{python_path}" if python_path else str(REPO_ROOT)
        )
        if env:
            subprocess_env.update(env)

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            env=subprocess_env,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout.strip())

    def test_auto_init_can_be_disabled(self):
        code = textwrap.dedent(
            """
            import json
            import lllm
            print(json.dumps({"discovery_done": lllm.get_default_runtime().discovery_done}))
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self._run_python(
                code,
                env={"LLLM_AUTO_INIT": "0"},
                cwd=Path(tmp_dir),
            )
        self.assertFalse(result["discovery_done"])

    def test_auto_init_does_not_load_shared_packages_when_disabled(self):
        code = textwrap.dedent(
            """
            import json
            import lllm
            print(json.dumps({
                "discovery_done": lllm.get_default_runtime().discovery_done,
                "packages": sorted(lllm.get_default_runtime().packages),
            }))
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shared_pkg_dir = tmp_path / "lllm_packages" / "pkg_alpha"
            shutil.copytree(CASES_DIR / "pkg_alpha", shared_pkg_dir)
            result = self._run_python(
                code,
                env={
                    "LLLM_AUTO_CWD_FALLBACK": "0",
                    "LLLM_AUTO_SHARED_PACKAGES": "0",
                },
                cwd=tmp_path,
            )
        self.assertTrue(result["discovery_done"])
        self.assertEqual(result["packages"], [])

    def test_invokers_module_keeps_litellm_lazy(self):
        code = textwrap.dedent(
            """
            import json
            import sys
            import lllm.invokers
            print(json.dumps({
                "litellm_loaded": "lllm.invokers.litellm" in sys.modules
            }))
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self._run_python(code, cwd=Path(tmp_dir))
        self.assertFalse(result["litellm_loaded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
