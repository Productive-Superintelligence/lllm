import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def build_docs(tmp_path: Path) -> Path:
    pytest.importorskip("mkdocs")
    site_dir = tmp_path / "site"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    return site_dir


def test_docs_render_mermaid_as_diagram_containers(tmp_path):
    site_dir = build_docs(tmp_path)
    html_pages = [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(site_dir.rglob("*.html"))
    ]
    diagram_pages = [
        path for path, html in html_pages if 'class="mermaid"' in html
    ]
    highlighted_pages = [
        path
        for path, html in html_pages
        if "language-mermaid" in html or "highlight-mermaid" in html
    ]

    assert diagram_pages
    assert not highlighted_pages

    for path in diagram_pages:
        html = path.read_text(encoding="utf-8")
        assert "javascripts/mermaid.js" in html
        assert "mermaid.min.js" in html


def test_docs_keep_light_brand_styles(tmp_path):
    site_dir = build_docs(tmp_path)
    custom_css = (site_dir / "stylesheets" / "custom.css").read_text(
        encoding="utf-8"
    )
    mermaid_js = (site_dir / "javascripts" / "mermaid.js").read_text(
        encoding="utf-8"
    )
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")

    assert ".md-header," in custom_css
    assert ".md-tabs {" in custom_css
    assert "background-color: #ffffff;" in custom_css
    assert ".psi-footer-brand img" in custom_css
    assert "height: 1.35rem;" in custom_css
    assert ".psi-brand img" in custom_css
    assert "width: min(10.5rem, 58vw);" in custom_css
    assert 'fontFamily: "Roboto, sans-serif"' in mermaid_js
    assert "assets/logo.svg" in index_html
    assert "assets/lllm-logo-text-dark.png" in index_html
