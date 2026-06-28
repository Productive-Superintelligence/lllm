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
        assert "javascripts/vendor/mermaid.min.js" in html
        assert "javascripts/mermaid.js" in html
        assert "cdn.jsdelivr" not in html
        assert "unpkg" not in html

    vendor_js = site_dir / "javascripts" / "vendor" / "mermaid.min.js"
    vendor_license = site_dir / "javascripts" / "vendor" / "mermaid-LICENSE.txt"
    assert vendor_js.exists()
    assert vendor_license.exists()


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
    assert ".md-header--shadow" in custom_css
    assert "background-color: #ffffff;" in custom_css
    assert "--md-footer-fg-color--light: var(--psi-ink);" in custom_css
    assert "--md-text-font-family" in custom_css
    assert ".md-header__button.md-logo" in custom_css
    assert "width: 1.45rem;" in custom_css
    assert ".md-nav__button.md-logo" in custom_css
    assert ".psi-footer-mark" in custom_css
    assert 'background-image: url("../assets/logo.svg");' in custom_css
    assert "font-size: 0.8rem;" in custom_css
    assert "height: 1.45rem;" in custom_css
    assert ".psi-brand img" in custom_css
    assert "height: clamp(2.65rem, 7vw, 3.25rem);" in custom_css
    assert ".md-typeset .mermaid svg" in custom_css
    assert "max-width: 100%;" in custom_css
    assert "min-width: 34rem;" in custom_css
    assert 'fontFamily: "Roboto, Helvetica Neue, Arial, sans-serif"' in mermaid_js
    assert "window.mermaid.startOnLoad = false" in mermaid_js
    assert 'securityLevel: "strict"' in mermaid_js
    assert "flowchart:" in mermaid_js
    assert "useMaxWidth: true" in mermaid_js
    assert "requestAnimationFrame" in mermaid_js
    assert "window.document$.subscribe(scheduleRender)" in mermaid_js
    assert "assets/logo.svg" in index_html
    assert "assets/lllm-logo-text-dark.png" in index_html
    assert "psi-footer-mark" in index_html
    assert 'src="/assets/lllm-logo-text-dark.png"' not in index_html

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert '<img src="assets/lllm-logo-text-dark.png" alt="LLLM" height="56">' in readme


def test_service_api_reference_matches_error_envelope():
    reference = (ROOT / "docs" / "reference" / "service-api.md").read_text(
        encoding="utf-8"
    )

    assert '"detail": {' in reference
    assert '"error": {' in reference
    assert '"type": "SchemaError"' in reference
    assert '"metadata": {}' in reference
    assert '"type": "TacticInputError"' not in reference


def test_public_text_does_not_use_staging_name():
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))
    text_paths.extend(
        path
        for path in (ROOT / "examples").rglob("*")
        if path.suffix in {".md", ".py", ".toml", ".yaml", ".yml"}
    )

    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        assert "LLLM v2" not in text, path
        assert "lllmv2" not in text, path
