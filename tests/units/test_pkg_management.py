"""
Tests for package management: install_package, export_package,
list_packages, remove_package, and the CLI ``lllm pkg`` commands.

All tests use isolated temporary directories so they never touch the user's
real ~/.lllm/packages or any project lllm_packages directory.
"""
from __future__ import annotations

import io
import sys
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

CASES_DIR = Path(__file__).parent.parent / "test_cases" / "packages"
PKG_SHAREABLE = CASES_DIR / "pkg_shareable"
PKG_SHARED_DEP = CASES_DIR / "pkg_shared_dep"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_rt():
    from lllm.core.runtime import Runtime
    return Runtime()


def _load(pkg_dir: Path, rt=None):
    from lllm.core.config import load_package
    if rt is None:
        rt = _fresh_rt()
    load_package(str(pkg_dir / "lllm.toml"), runtime=rt)
    return rt


def _make_zip(pkg_dir: Path, tmp_dir: Path, zip_name: str | None = None) -> Path:
    """Create a zip of a test-case package, mirroring export_package output."""
    pkg_name = pkg_dir.name
    zip_path = tmp_dir / (zip_name or f"{pkg_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(pkg_dir)
            if any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                continue
            zf.write(f, Path(pkg_name) / rel)
    return zip_path


# ===========================================================================
# export_package
# ===========================================================================

class TestExportPackage(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.rt = _load(PKG_SHAREABLE)

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_creates_zip(self):
        from lllm.core.runtime import export_package
        out = export_package("pkg_shareable", self.tmp / "out.zip", runtime=self.rt)
        self.assertTrue(out.exists())
        self.assertEqual(out.suffix, ".zip")

    def test_export_auto_adds_zip_suffix(self):
        from lllm.core.runtime import export_package
        out = export_package("pkg_shareable", self.tmp / "out", runtime=self.rt)
        self.assertEqual(out.suffix, ".zip")

    def test_export_zip_contains_lllm_toml(self):
        from lllm.core.runtime import export_package
        out = export_package("pkg_shareable", self.tmp / "out.zip", runtime=self.rt)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        self.assertIn("pkg_shareable/lllm.toml", names)

    def test_export_zip_contains_prompts(self):
        from lllm.core.runtime import export_package
        out = export_package("pkg_shareable", self.tmp / "out.zip", runtime=self.rt)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        self.assertTrue(any("prompts/main.py" in n for n in names))

    def test_export_excludes_pycache(self):
        from lllm.core.runtime import export_package
        out = export_package("pkg_shareable", self.tmp / "out.zip", runtime=self.rt)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        self.assertFalse(any("__pycache__" in n for n in names))

    def test_export_unknown_package_raises(self):
        from lllm.core.runtime import export_package
        with self.assertRaises(ValueError):
            export_package("no_such_pkg", self.tmp / "x.zip", runtime=self.rt)

    def test_export_bundle_deps_includes_dep(self):
        from lllm.core.runtime import export_package
        # Load both packages so the runtime knows about pkg_shared_dep
        rt = _load(PKG_SHAREABLE)
        out = export_package(
            "pkg_shareable", self.tmp / "bundled.zip",
            bundle_deps=True, runtime=rt,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        self.assertTrue(
            any("lllm_packages/pkg_shared_dep/lllm.toml" in n for n in names),
            f"Expected bundled dep lllm.toml, got: {names}",
        )

    def test_export_bundle_deps_false_no_dep_folder(self):
        from lllm.core.runtime import export_package
        out = export_package(
            "pkg_shareable", self.tmp / "no_bundle.zip",
            bundle_deps=False, runtime=self.rt,
        )
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        self.assertFalse(any("lllm_packages" in n for n in names))


# ===========================================================================
# install_package
# ===========================================================================

class TestInstallPackage(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.install_root = self.tmp / "packages"

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, zip_path, **kwargs):
        from lllm.core.runtime import install_package
        rt = _fresh_rt()
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.tmp),
        ):
            return install_package(zip_path, load=False, **kwargs), rt

    def test_install_creates_directory(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        dest, _ = self._install(zip_path)
        self.assertTrue(dest.is_dir())
        self.assertTrue((dest / "lllm.toml").is_file())

    def test_install_default_scope_user(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        dest, _ = self._install(zip_path)
        # Should land in <home>/.lllm/packages/pkg_shareable
        self.assertIn(".lllm", str(dest))
        self.assertEqual(dest.name, "pkg_shareable")

    def test_install_with_alias_renames_folder(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        dest, _ = self._install(zip_path, alias="my-share")
        self.assertEqual(dest.name, "my-share")

    def test_install_with_alias_patches_toml_name(self):
        import tomllib
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        dest, _ = self._install(zip_path, alias="my-share")
        with (dest / "lllm.toml").open("rb") as fh:
            data = tomllib.load(fh)
        self.assertEqual(data["package"]["name"], "my-share")

    def test_install_collision_raises(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with patch("lllm.core.runtime.Path.home", return_value=self.tmp):
            install_package(zip_path, load=False)
            with self.assertRaises(FileExistsError):
                install_package(zip_path, load=False)

    def test_install_collision_with_alias_raises(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with patch("lllm.core.runtime.Path.home", return_value=self.tmp):
            install_package(zip_path, alias="dup-alias", load=False)
            with self.assertRaises(FileExistsError):
                install_package(zip_path, alias="dup-alias", load=False)

    def test_install_missing_zip_raises(self):
        from lllm.core.runtime import install_package
        with self.assertRaises(FileNotFoundError):
            install_package(self.tmp / "nonexistent.zip", load=False)

    def test_install_invalid_alias_raises(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with self.assertRaises(ValueError):
            install_package(zip_path, alias="INVALID ALIAS!", load=False)

    def test_install_zip_without_toml_raises(self):
        from lllm.core.runtime import install_package
        bad_zip = self.tmp / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("some_folder/random.txt", "no lllm.toml here")
        with self.assertRaises(ValueError):
            install_package(bad_zip, load=False)

    def test_install_loads_into_runtime(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        rt = _fresh_rt()
        with patch("lllm.core.runtime.Path.home", return_value=self.tmp):
            install_package(zip_path, load=True, runtime=rt)
        self.assertIn("pkg_shareable", rt.packages)

    def test_install_scope_project(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        project_root = self.tmp / "myproject"
        project_root.mkdir()
        (project_root / "lllm.toml").write_text(
            '[package]\nname = "myproject"\n', encoding="utf-8"
        )
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.tmp),
            patch("lllm.core.config.find_config_file",
                  return_value=project_root / "lllm.toml"),
        ):
            dest = install_package(zip_path, scope="project", load=False)
        self.assertIn("lllm_packages", str(dest))

    def test_install_invalid_scope_raises(self):
        from lllm.core.runtime import install_package
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with self.assertRaises(ValueError):
            install_package(zip_path, scope="galactic", load=False)


# ===========================================================================
# list_packages
# ===========================================================================

class TestListPackages(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _populate_user_packages(self, *pkg_dirs: Path) -> None:
        """Copy minimal pkg structure into the mocked user packages dir."""
        import shutil
        user_pkgs = self.home / ".lllm" / "packages"
        user_pkgs.mkdir(parents=True, exist_ok=True)
        for pkg_dir in pkg_dirs:
            shutil.copytree(str(pkg_dir), str(user_pkgs / pkg_dir.name))

    def test_lists_user_packages(self):
        from lllm.core.runtime import list_packages
        self._populate_user_packages(PKG_SHAREABLE, PKG_SHARED_DEP)
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(runtime=_fresh_rt())
        names = [p["name"] for p in pkgs]
        self.assertIn("pkg_shareable", names)
        self.assertIn("pkg_shared_dep", names)

    def test_scope_user_only(self):
        from lllm.core.runtime import list_packages
        self._populate_user_packages(PKG_SHAREABLE)
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(scope="user", runtime=_fresh_rt())
        self.assertTrue(all(p["scope"] == "user" for p in pkgs))

    def test_scope_project_only(self):
        from lllm.core.runtime import list_packages
        project_root = self.tmp / "proj"
        project_root.mkdir()
        import shutil
        shutil.copytree(str(PKG_SHARED_DEP),
                        str(project_root / "lllm_packages" / PKG_SHARED_DEP.name))
        proj_toml = project_root / "lllm.toml"
        proj_toml.write_text('[package]\nname = "proj"\n', encoding="utf-8")
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=proj_toml),
        ):
            pkgs = list_packages(scope="project", runtime=_fresh_rt())
        self.assertTrue(all(p["scope"] == "project" for p in pkgs))
        names = [p["name"] for p in pkgs]
        self.assertIn("pkg_shared_dep", names)

    def test_empty_returns_runtime_packages(self):
        from lllm.core.runtime import list_packages
        rt = _load(PKG_SHAREABLE)
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(runtime=rt)
        names = [p["name"] for p in pkgs]
        # Runtime-loaded packages appear with scope "unknown"
        self.assertIn("pkg_shareable", names)

    def test_metadata_fields_present(self):
        from lllm.core.runtime import list_packages
        self._populate_user_packages(PKG_SHAREABLE)
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(runtime=_fresh_rt())
        pkg = next(p for p in pkgs if p["name"] == "pkg_shareable")
        self.assertEqual(pkg["version"], "1.2.0")
        self.assertIn("scope", pkg)
        self.assertIn("path", pkg)

    def test_no_packages_returns_empty(self):
        from lllm.core.runtime import list_packages
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(runtime=_fresh_rt())
        self.assertEqual(pkgs, [])


# ===========================================================================
# remove_package
# ===========================================================================

class TestRemovePackage(unittest.TestCase):

    def setUp(self):
        import shutil, tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.user_pkgs = self.home / ".lllm" / "packages"
        self.user_pkgs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(PKG_SHAREABLE), str(self.user_pkgs / "pkg_shareable"))
        shutil.copytree(str(PKG_SHARED_DEP), str(self.user_pkgs / "pkg_shared_dep"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_remove_deletes_directory(self):
        from lllm.core.runtime import remove_package
        target = self.user_pkgs / "pkg_shareable"
        self.assertTrue(target.is_dir())
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            remove_package("pkg_shareable", runtime=_fresh_rt())
        self.assertFalse(target.exists())

    def test_remove_returns_deleted_path(self):
        from lllm.core.runtime import remove_package
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            deleted = remove_package("pkg_shareable", runtime=_fresh_rt())
        self.assertEqual(deleted.name, "pkg_shareable")

    def test_remove_unregisters_from_runtime(self):
        from lllm.core.runtime import remove_package
        rt = _fresh_rt()
        from lllm.core.config import load_package
        load_package(str(self.user_pkgs / "pkg_shareable" / "lllm.toml"), runtime=rt)
        self.assertIn("pkg_shareable", rt.packages)
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            remove_package("pkg_shareable", runtime=rt)
        self.assertNotIn("pkg_shareable", rt.packages)

    def test_remove_nonexistent_raises(self):
        from lllm.core.runtime import remove_package
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            with self.assertRaises(FileNotFoundError):
                remove_package("no_such_pkg", runtime=_fresh_rt())

    def test_remove_scope_user(self):
        from lllm.core.runtime import remove_package
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            remove_package("pkg_shared_dep", scope="user", runtime=_fresh_rt())
        self.assertFalse((self.user_pkgs / "pkg_shared_dep").exists())

    def test_remove_in_multiple_scopes_raises(self):
        """If a package exists in both user and project scope, must specify scope."""
        import shutil
        from lllm.core.runtime import remove_package
        project_root = self.tmp / "proj"
        project_root.mkdir()
        proj_pkgs = project_root / "lllm_packages"
        shutil.copytree(str(PKG_SHAREABLE), str(proj_pkgs / "pkg_shareable"))
        proj_toml = project_root / "lllm.toml"
        proj_toml.write_text('[package]\nname = "proj"\n', encoding="utf-8")
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=proj_toml),
        ):
            with self.assertRaises(ValueError):
                remove_package("pkg_shareable", runtime=_fresh_rt())


# ===========================================================================
# Full round-trip: export → install → list → remove
# ===========================================================================

class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_then_install_then_list_then_remove(self):
        from lllm.core.runtime import (
            export_package, install_package, list_packages, remove_package,
        )

        # 1. Load and export
        rt = _load(PKG_SHAREABLE)
        zip_path = export_package("pkg_shareable", self.tmp / "shareable.zip", runtime=rt)
        self.assertTrue(zip_path.exists())

        # 2. Install into mocked user home
        with patch("lllm.core.runtime.Path.home", return_value=self.home):
            dest = install_package(str(zip_path), load=False)
        self.assertTrue(dest.is_dir())
        self.assertEqual(dest.name, "pkg_shareable")

        # 3. List — should appear
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs = list_packages(runtime=_fresh_rt())
        names = [p["name"] for p in pkgs]
        self.assertIn("pkg_shareable", names)

        # 4. Remove
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            remove_package("pkg_shareable", runtime=_fresh_rt())
        self.assertFalse(dest.exists())

        # 5. List again — should be gone
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
        ):
            pkgs2 = list_packages(runtime=_fresh_rt())
        names2 = [p["name"] for p in pkgs2]
        self.assertNotIn("pkg_shareable", names2)

    def test_export_bundle_deps_then_install(self):
        """Bundled-dep export should unpack with deps inside lllm_packages/."""
        from lllm.core.runtime import export_package, install_package

        rt = _load(PKG_SHAREABLE)
        zip_path = export_package(
            "pkg_shareable", self.tmp / "bundled.zip",
            bundle_deps=True, runtime=rt,
        )

        with patch("lllm.core.runtime.Path.home", return_value=self.home):
            dest = install_package(str(zip_path), load=False)

        dep_dir = dest / "lllm_packages" / "pkg_shared_dep"
        self.assertTrue(dep_dir.is_dir(), f"Expected bundled dep at {dep_dir}")
        self.assertTrue((dep_dir / "lllm.toml").is_file())


# ===========================================================================
# _collect_package_deps
# ===========================================================================

class TestCollectPackageDeps(unittest.TestCase):

    def test_collects_direct_dep(self):
        from lllm.core.runtime import _collect_package_deps
        rt = _load(PKG_SHAREABLE)
        deps = _collect_package_deps("pkg_shareable", rt)
        self.assertIn("pkg_shared_dep", deps)

    def test_excludes_self(self):
        from lllm.core.runtime import _collect_package_deps
        rt = _load(PKG_SHAREABLE)
        deps = _collect_package_deps("pkg_shareable", rt)
        self.assertNotIn("pkg_shareable", deps)

    def test_no_deps_returns_empty(self):
        from lllm.core.runtime import _collect_package_deps
        rt = _load(PKG_SHARED_DEP)
        deps = _collect_package_deps("pkg_shared_dep", rt)
        self.assertEqual(deps, {})

    def test_transitive_deps(self):
        """pkg_chain_a → chain_b → chain_c → chain_d; all should be collected."""
        from lllm.core.runtime import _collect_package_deps
        from lllm.core.config import load_package
        chain_a = CASES_DIR / "pkg_chain_a"
        rt = _fresh_rt()
        load_package(str(chain_a / "lllm.toml"), runtime=rt)
        deps = _collect_package_deps("pkg_chain_a", rt)
        self.assertIn("pkg_chain_b", deps)
        self.assertIn("pkg_chain_c", deps)
        self.assertIn("pkg_chain_d", deps)


# ===========================================================================
# CLI: lllm pkg
# ===========================================================================

def _run_cli(*args):
    """Invoke cli.main() with the given args, return (stdout, stderr, exit_code)."""
    import io
    from lllm.cli import main
    old_argv = sys.argv
    sys.argv = ["lllm"] + list(args)
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    exit_code = 0
    try:
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            main()
    except SystemExit as e:
        exit_code = e.code or 0
    finally:
        sys.argv = old_argv
    return out_buf.getvalue(), err_buf.getvalue(), exit_code


class TestCliPkg(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_pkg_no_subcommand_exits_nonzero(self):
        _, err, code = _run_cli("pkg")
        self.assertNotEqual(code, 0)
        self.assertIn("install", err)

    def test_cli_install(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with patch("lllm.core.runtime.Path.home", return_value=self.home):
            out, err, code = _run_cli("pkg", "install", str(zip_path), "--no-load")
        self.assertEqual(code, 0, err)
        self.assertIn("Installed", out)
        self.assertTrue((self.home / ".lllm" / "packages" / "pkg_shareable").is_dir())

    def test_cli_install_with_alias(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with patch("lllm.core.runtime.Path.home", return_value=self.home):
            out, err, code = _run_cli(
                "pkg", "install", str(zip_path), "--alias", "my-share", "--no-load"
            )
        self.assertEqual(code, 0, err)
        self.assertIn("my-share", out)
        self.assertTrue((self.home / ".lllm" / "packages" / "my-share").is_dir())

    def test_cli_install_duplicate_exits_nonzero(self):
        zip_path = _make_zip(PKG_SHAREABLE, self.tmp)
        with patch("lllm.core.runtime.Path.home", return_value=self.home):
            _run_cli("pkg", "install", str(zip_path), "--no-load")
            _, err, code = _run_cli("pkg", "install", str(zip_path), "--no-load")
        self.assertNotEqual(code, 0)

    def test_cli_list_empty(self):
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
            patch("lllm.core.runtime.get_default_runtime", return_value=_fresh_rt()),
        ):
            out, _, code = _run_cli("pkg", "list")
        self.assertEqual(code, 0)
        self.assertIn("No packages", out)

    def test_cli_list_shows_installed(self):
        import shutil
        user_pkgs = self.home / ".lllm" / "packages"
        user_pkgs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(PKG_SHAREABLE), str(user_pkgs / "pkg_shareable"))
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
            patch("lllm.core.runtime.get_default_runtime", return_value=_fresh_rt()),
        ):
            out, _, code = _run_cli("pkg", "list")
        self.assertEqual(code, 0)
        self.assertIn("pkg_shareable", out)
        self.assertIn("1.2.0", out)

    def test_cli_remove(self):
        import shutil
        user_pkgs = self.home / ".lllm" / "packages"
        user_pkgs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(PKG_SHAREABLE), str(user_pkgs / "pkg_shareable"))
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
            patch("lllm.core.runtime.get_default_runtime", return_value=_fresh_rt()),
        ):
            out, err, code = _run_cli("pkg", "remove", "pkg_shareable")
        self.assertEqual(code, 0, err)
        self.assertIn("Removed", out)
        self.assertFalse((user_pkgs / "pkg_shareable").exists())

    def test_cli_remove_nonexistent_exits_nonzero(self):
        with (
            patch("lllm.core.runtime.Path.home", return_value=self.home),
            patch("lllm.core.config.find_config_file", return_value=None),
            patch("lllm.core.runtime.get_default_runtime", return_value=_fresh_rt()),
        ):
            _, err, code = _run_cli("pkg", "remove", "no_such_pkg")
        self.assertNotEqual(code, 0)

    def test_cli_export(self):
        rt = _load(PKG_SHAREABLE)
        out_path = self.tmp / "export_out.zip"
        with patch("lllm.core.runtime.get_default_runtime", return_value=rt):
            # load_runtime is called inside _cmd_export; patch it to return rt
            with patch("lllm.load_runtime", return_value=rt):
                out, err, code = _run_cli("pkg", "export", "pkg_shareable", str(out_path))
        self.assertEqual(code, 0, err)
        self.assertIn("Exported", out)
        self.assertTrue(out_path.exists())

    def test_cli_export_bundle_deps(self):
        rt = _load(PKG_SHAREABLE)
        out_path = self.tmp / "export_bundled.zip"
        with (
            patch("lllm.core.runtime.get_default_runtime", return_value=rt),
            patch("lllm.load_runtime", return_value=rt),
        ):
            out, err, code = _run_cli(
                "pkg", "export", "pkg_shareable", str(out_path), "--bundle-deps"
            )
        self.assertEqual(code, 0, err)
        self.assertIn("bundled", out)
        with zipfile.ZipFile(out_path) as zf:
            names = zf.namelist()
        self.assertTrue(any("pkg_shared_dep" in n for n in names))

    def test_cli_export_unknown_pkg_exits_nonzero(self):
        fresh_rt = _fresh_rt()
        with (
            patch("lllm.core.runtime.get_default_runtime", return_value=fresh_rt),
            patch("lllm.load_runtime", return_value=fresh_rt),
        ):
            _, err, code = _run_cli("pkg", "export", "ghost_pkg", str(self.tmp / "x.zip"))
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
