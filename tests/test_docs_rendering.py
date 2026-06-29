import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

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


def test_docs_render_mermaid_svgs_in_chromium(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )
    if not chromium:
        pytest.skip("Chromium executable is not available")

    site_dir = build_docs(tmp_path)
    diagram_pages = [
        path
        for path in sorted(site_dir.rglob("*.html"))
        if 'class="mermaid"' in path.read_text(encoding="utf-8")
    ]

    assert diagram_pages

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--no-sandbox"],
        )
        for path in diagram_pages:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type in {"error", "warning"}
                else None,
            )
            page.goto(path.as_uri(), wait_until="domcontentloaded")
            page.wait_for_function(
                """
                () => {
                  const diagrams = document.querySelectorAll('.mermaid');
                  return diagrams.length > 0 &&
                    document.querySelectorAll('.mermaid svg').length === diagrams.length;
                }
                """,
                timeout=5000,
            )
            assert page.evaluate(
                "() => document.querySelectorAll('[data-mermaid-error=\"true\"]').length"
            ) == 0
            assert page.evaluate(
                "() => document.querySelectorAll('code.language-mermaid, code.highlight-mermaid').length"
            ) == 0
            assert console_errors == []
            page.close()
        browser.close()


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
    assert '--md-text-font: "Inter";' in custom_css
    assert '--md-code-font: "JetBrains Mono";' in custom_css
    assert "--md-text-font-family" in custom_css
    assert '"JetBrains Mono", SFMono-Regular' in custom_css
    assert "-apple-system" in custom_css
    assert "--psi-brand-width: 26rem;" in custom_css
    assert ".md-header__button.md-logo" in custom_css
    assert "width: 1.2rem;" in custom_css
    assert ".md-search__form .md-icon svg" in custom_css
    assert "fill: currentcolor;" in custom_css
    assert ".md-nav__button.md-logo" in custom_css
    assert ".psi-footer-mark" in custom_css
    assert 'background-image: url("../assets/logo.svg");' in custom_css
    assert "font-size: 0.8rem;" in custom_css
    assert "min-height: 2.2rem;" in custom_css
    assert ".md-social__link" in custom_css
    assert "height: 1rem;" in custom_css
    assert ".psi-brand img" in custom_css
    assert "width: min(var(--psi-brand-width), 100%);" in custom_css
    assert 'img[src$="#only-dark"]' in custom_css
    assert ".md-typeset .mermaid svg" in custom_css
    assert "max-width: 100%;" in custom_css
    assert "min-width: 34rem;" in custom_css
    assert ".md-typeset .mermaid .node rect" in custom_css
    assert ".md-typeset .mermaid .edgePath path" in custom_css
    assert ".md-typeset .mermaid marker path" in custom_css
    assert "var fontFamily" in mermaid_js
    assert "Inter, -apple-system, BlinkMacSystemFont" in mermaid_js
    assert "window.mermaid.startOnLoad = false" in mermaid_js
    assert 'securityLevel: "strict"' in mermaid_js
    assert "flowchart:" in mermaid_js
    assert "htmlLabels: false" in mermaid_js
    assert "useMaxWidth: true" in mermaid_js
    assert "data-mermaid-source" in mermaid_js
    assert "normalizeDiagramNode" in mermaid_js
    assert "pre code.language-mermaid" in mermaid_js
    assert "data-mermaid-error" in mermaid_js
    assert "renderWithRun" in mermaid_js
    assert "renderNodeFallback" in mermaid_js
    assert "window.mermaid.render" in mermaid_js
    assert "attempt < 30" in mermaid_js
    assert "requestAnimationFrame" in mermaid_js
    assert "window.document$.subscribe(scheduleRender)" in mermaid_js
    assert 'window.addEventListener("load", scheduleRender)' in mermaid_js
    assert "assets/logo.svg" in index_html
    assert "assets/lllm-logo-text-dark.png#only-light" in index_html
    assert "assets/lllm-logo-text-white.png#only-dark" in index_html
    assert "psi-footer-mark" in index_html
    assert "<div class=\"md-source__repository\">\n    GitHub\n  </div>" in index_html
    assert 'data-md-component="source"' not in index_html
    assert 'src="/assets/lllm-logo-text-dark.png"' not in index_html

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert '<p align="center">' in readme
    assert '<img src="assets/lllm-logo-text-dark.png" alt="LLLM" width="560">' in readme
    assert (site_dir / "CNAME").read_text(encoding="utf-8").strip() == "lllm.one"


def test_service_api_reference_matches_error_envelope():
    reference = (ROOT / "docs" / "reference" / "service-api.md").read_text(
        encoding="utf-8"
    )

    assert '"detail": {' in reference
    assert '"error": {' in reference
    assert '"type": "SchemaError"' in reference
    assert '"metadata": {}' in reference
    assert '"type": "TacticInputError"' not in reference


def test_composition_guide_documents_mixed_local_config():
    guide = (ROOT / "docs" / "guides" / "composition.md").read_text(
        encoding="utf-8"
    )

    assert "TacticResolver.from_config()" in guide
    assert "shared `.psi/config.toml`" in guide
    assert "non-tactic refs" in guide
    assert "psi://demo/echo/services/api" in guide
    assert "psi://demo/echo/channels/events" in guide
    assert "Malformed tactic refs still fail validation." in guide


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
        assert "SSSN v2" not in text, path
        assert "sssnv2" not in text, path


def test_tutorials_keep_step_by_step_shape():
    required = [
        "Goal:",
        "## Prerequisites",
        "## Files Used",
        "## Verify",
        "Expected output:",
        "Next,",
    ]

    for path in sorted((ROOT / "docs" / "tutorials").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for marker in required:
            assert marker in text, f"{path.relative_to(ROOT)} missing {marker}"


def test_readme_and_docs_local_links_resolve():
    pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))
    skip_prefixes = ("http://", "https://", "mailto:", "#")

    missing = []
    for path in sorted(text_paths):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).strip()
            if not target or target.startswith(skip_prefixes) or "://" in target:
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            if not (path.parent / unquote(target)).resolve().exists():
                missing.append(f"{path.relative_to(ROOT)} -> {match.group(1)}")

    assert missing == []
