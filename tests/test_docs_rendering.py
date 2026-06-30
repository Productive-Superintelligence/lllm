import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]


def chromium_executable() -> str | None:
    return (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )


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


def css_block(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", css, re.S)
    assert match, f"missing CSS selector: {selector}"
    return match.group("body")


def test_docs_use_tactic_protocol_framing():
    sources = {
        "README": ROOT / "README.md",
        "docs home": ROOT / "docs" / "index.md",
        "package docstring": ROOT / "lllm" / "__init__.py",
        "package metadata": ROOT / "pyproject.toml",
        "site metadata": ROOT / "mkdocs.yml",
    }
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sources.values()
    )

    assert "protocol and service layer for reusable agentic tactics" in combined
    assert "Protocol-first tactic services" not in combined
    assert "protocol-first service infrastructure" not in combined


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
        assert "javascripts/mermaid-init.20260629.js" in html
        assert "cdn.jsdelivr" not in html
        assert "unpkg" not in html

    vendor_js = site_dir / "javascripts" / "vendor" / "mermaid.min.js"
    vendor_license = site_dir / "javascripts" / "vendor" / "mermaid-LICENSE.txt"
    assert vendor_js.exists()
    assert vendor_license.exists()


def test_docs_render_mermaid_svgs_in_chromium(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = chromium_executable()
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
                  const diagrams = [...document.querySelectorAll('.md-typeset .mermaid')];
                  return diagrams.length > 0 &&
                    diagrams.every((diagram) =>
                      diagram.hasAttribute('data-mermaid-source') &&
                      diagram.querySelector('svg')
                    );
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
            assert page.evaluate("() => document.body.innerText.includes('flowchart')") is False
            diagram_metrics = page.evaluate(
                """
                () => [...document.querySelectorAll(".md-typeset .mermaid")]
                  .map((diagram) => {
                  const svg = diagram.querySelector("svg");
                  const label = svg.querySelector(
                    "foreignObject span, .nodeLabel, .edgeLabel, text, tspan"
                  );
                  const labelStyle = label ? getComputedStyle(label) : null;
                  const diagramRect = diagram.getBoundingClientRect();
                  const svgRect = svg.getBoundingClientRect();
                  const shapes = [
                    ...svg.querySelectorAll(
                      "path, rect, circle, ellipse, polygon, line"
                    )
                  ].map((element) => {
                    const style = getComputedStyle(element);
                    return {
                      fill: style.fill,
                      stroke: style.stroke,
                    };
                  });
                  return {
                    diagramHeight: diagramRect.height,
                    diagramWidth: diagramRect.width,
                    labelColor: labelStyle ? labelStyle.color : "",
                    labelFill: labelStyle ? labelStyle.fill : "",
                    shapeInk: shapes.filter((shape) =>
                      [shape.fill, shape.stroke].includes("rgb(5, 5, 5)")
                    ).length,
                    source: diagram.getAttribute("data-mermaid-source") || "",
                    svgHeight: svgRect.height,
                    svgText: svg.textContent.trim(),
                    svgWidth: svgRect.width,
                  };
                })
                """
            )
            for metric in diagram_metrics:
                assert metric["source"].lstrip().startswith("flowchart")
                assert metric["diagramWidth"] > 120
                assert metric["diagramHeight"] > 80
                assert metric["svgWidth"] > 120
                assert metric["svgHeight"] > 80
                assert metric["svgText"]
                assert metric["shapeInk"] > 0
                assert "rgb(5, 5, 5)" in {
                    metric["labelColor"],
                    metric["labelFill"],
                }
            assert console_errors == []
            page.close()
        browser.close()


def test_docs_chrome_matches_light_visual_contract(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = chromium_executable()
    if not chromium:
        pytest.skip("Chromium executable is not available")

    site_dir = build_docs(tmp_path)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(
            (site_dir / "index.html").as_uri(), wait_until="domcontentloaded"
        )
        page.wait_for_selector(
            ".md-header__button.md-logo img, .md-header__button.md-logo svg"
        )
        metrics = page.evaluate(
            """
            () => {
              const inspect = (selector) => {
                const element = document.querySelector(selector);
                if (!element) {
                  return null;
                }
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return {
                  backgroundColor: style.backgroundColor,
                  boxShadow: style.boxShadow,
                  color: style.color,
                  display: style.display,
                  fontFamily: style.fontFamily,
                  fontWeight: style.fontWeight,
                  height: rect.height,
                  src: element.getAttribute("src") || "",
                  width: rect.width,
                };
              };
              const brandImages = [...document.querySelectorAll(".psi-brand img")]
                .map((element) => {
                  const style = getComputedStyle(element);
                  const rect = element.getBoundingClientRect();
                  return {
                    display: style.display,
                    height: rect.height,
                    src: element.getAttribute("src") || "",
                    width: rect.width,
                  };
                });
              return {
                bodyFont: getComputedStyle(document.body)
                  .getPropertyValue("--md-text-font-family")
                  .trim(),
                codeFont: getComputedStyle(document.body)
                  .getPropertyValue("--md-code-font-family")
                  .trim(),
                footer: inspect(".md-footer-meta"),
                footerMark: inspect(".psi-footer-wordmark"),
                header: inspect(".md-header"),
                headerLogo: inspect(
                  ".md-header__button.md-logo img, .md-header__button.md-logo svg"
                ),
                palette: inspect(".md-header__option[data-md-component='palette']"),
                tabs: inspect(".md-tabs"),
                title: inspect(".md-header__title"),
                brandImages,
              };
            }
            """
        )
        page.close()
        browser.close()

    assert metrics["header"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["tabs"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["footer"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["header"]["color"] == "rgb(5, 5, 5)"
    assert metrics["footer"]["color"] == "rgb(5, 5, 5)"
    assert metrics["header"]["boxShadow"] == "none"
    assert metrics["title"]["fontWeight"] == "700"
    assert metrics["headerLogo"]["width"] == pytest.approx(24, abs=1)
    assert metrics["headerLogo"]["height"] == pytest.approx(24, abs=1)
    assert metrics["palette"]["width"] == pytest.approx(0, abs=1)
    assert metrics["palette"]["height"] == pytest.approx(0, abs=1)
    assert metrics["footer"]["height"] == pytest.approx(44, abs=1)
    assert metrics["footerMark"]["width"] == pytest.approx(100, abs=2)
    assert metrics["footerMark"]["height"] == pytest.approx(27, abs=2)
    assert "Roboto" in metrics["bodyFont"]
    assert "Roboto Mono" in metrics["codeFont"]

    visible_brands = [
        image for image in metrics["brandImages"] if image["display"] == "block"
    ]
    hidden_brands = [
        image for image in metrics["brandImages"] if image["display"] == "none"
    ]
    assert len(visible_brands) == 1
    assert visible_brands[0]["src"] == "assets/lllm-logo-text-dark.png#only-light"
    assert visible_brands[0]["width"] == pytest.approx(320, abs=3)
    assert visible_brands[0]["height"] == pytest.approx(90, abs=2)
    assert any(
        image["src"] == "assets/lllm-logo-text-white.png#only-dark"
        for image in hidden_brands
    )


def test_docs_mobile_chrome_keeps_visual_contract(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = chromium_executable()
    if not chromium:
        pytest.skip("Chromium executable is not available")

    site_dir = build_docs(tmp_path)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
        )
        page.goto(
            (site_dir / "index.html").as_uri(), wait_until="domcontentloaded"
        )
        page.wait_for_function(
            """
            () => {
              const diagrams = [...document.querySelectorAll('.md-typeset .mermaid')];
              return diagrams.length > 0 &&
                diagrams.every((diagram) =>
                  diagram.hasAttribute('data-mermaid-source') &&
                  diagram.querySelector('svg')
                );
            }
            """,
            timeout=5000,
        )
        metrics = page.evaluate(
            """
            () => {
              const inspect = (selector) => {
                const element = document.querySelector(selector);
                if (!element) {
                  return null;
                }
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return {
                  backgroundColor: style.backgroundColor,
                  color: style.color,
                  display: style.display,
                  height: rect.height,
                  src: element.getAttribute("src") || "",
                  width: rect.width,
                };
              };
              const images = (selector) =>
                [...document.querySelectorAll(selector)].map((element) => {
                  const style = getComputedStyle(element);
                  const rect = element.getBoundingClientRect();
                  return {
                    display: style.display,
                    height: rect.height,
                    src: element.getAttribute("src") || "",
                    width: rect.width,
                  };
                });
              return {
                bodyFont: getComputedStyle(document.body)
                  .getPropertyValue("--md-text-font-family")
                  .trim(),
                codeFont: getComputedStyle(document.body)
                  .getPropertyValue("--md-code-font-family")
                  .trim(),
                docWidth: document.documentElement.scrollWidth,
                viewportWidth: window.innerWidth,
                footer: inspect(".md-footer-meta"),
                footerMark: inspect(".psi-footer-wordmark"),
                header: inspect(".md-header"),
                headerLogo: inspect(
                  ".md-header__button.md-logo img, .md-header__button.md-logo svg"
                ),
                mermaid: inspect(".md-typeset .mermaid"),
                brandImages: images(".psi-brand img"),
                mermaidSvgs: images(".md-typeset .mermaid svg"),
              };
            }
            """
        )
        page.click('.md-header__button[for="__drawer"]')
        page.wait_for_timeout(350)
        drawer_metrics = page.evaluate(
            """
            () => {
              const inspect = (selector) => {
                const element = document.querySelector(selector);
                if (!element) {
                  return null;
                }
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return {
                  backgroundColor: style.backgroundColor,
                  color: style.color,
                  height: rect.height,
                  src: element.getAttribute("src") || "",
                  width: rect.width,
                };
              };
              return {
                logo: inspect(
                  ".md-nav__button.md-logo img, .md-nav__button.md-logo svg"
                ),
                title: inspect('.md-nav--primary .md-nav__title[for="__drawer"]'),
              };
            }
            """
        )
        page.close()
        browser.close()

    assert metrics["docWidth"] <= metrics["viewportWidth"] + 1
    assert metrics["header"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["footer"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["header"]["color"] == "rgb(5, 5, 5)"
    assert metrics["footer"]["color"] == "rgb(5, 5, 5)"
    assert metrics["header"]["height"] == pytest.approx(49, abs=1)
    assert metrics["headerLogo"]["src"] == "assets/logo.svg"
    assert metrics["headerLogo"]["width"] == pytest.approx(24, abs=1)
    assert metrics["headerLogo"]["height"] == pytest.approx(24, abs=1)
    assert metrics["footer"]["height"] <= 72
    assert metrics["footerMark"]["width"] == pytest.approx(100, abs=2)
    assert metrics["footerMark"]["height"] == pytest.approx(27, abs=2)
    assert "Roboto" in metrics["bodyFont"]
    assert "Roboto Mono" in metrics["codeFont"]
    assert metrics["mermaid"]["width"] <= metrics["viewportWidth"]
    assert metrics["mermaidSvgs"][0]["width"] > metrics["viewportWidth"]
    assert drawer_metrics["title"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert drawer_metrics["title"]["color"] == "rgb(5, 5, 5)"
    assert drawer_metrics["logo"]["src"] == "assets/logo.svg"
    assert drawer_metrics["logo"]["width"] == pytest.approx(48, abs=1)
    assert drawer_metrics["logo"]["height"] == pytest.approx(48, abs=1)

    visible_brands = [
        image for image in metrics["brandImages"] if image["display"] == "block"
    ]
    hidden_brands = [
        image for image in metrics["brandImages"] if image["display"] == "none"
    ]
    assert len(visible_brands) == 1
    assert visible_brands[0]["src"] == "assets/lllm-logo-text-dark.png#only-light"
    assert visible_brands[0]["width"] < metrics["viewportWidth"]
    assert 63 <= visible_brands[0]["height"] <= 67
    assert any(
        image["src"] == "assets/lllm-logo-text-white.png#only-dark"
        for image in hidden_brands
    )


def test_docs_keep_light_brand_styles(tmp_path):
    site_dir = build_docs(tmp_path)
    custom_css = (site_dir / "stylesheets" / "custom.20260629.css").read_text(
        encoding="utf-8"
    )
    mermaid_js = (site_dir / "javascripts" / "mermaid-init.20260629.js").read_text(
        encoding="utf-8"
    )
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    mermaid_svg_css = css_block(custom_css, ".md-typeset .mermaid svg")

    assert ".md-header," in custom_css
    assert ".md-tabs {" in custom_css
    assert ".md-header--shadow" in custom_css
    assert "background-color: #ffffff;" in custom_css
    assert "--md-footer-fg-color--light: var(--psi-ink);" in custom_css
    assert ".md-footer-meta__inner" in custom_css
    assert "display: flex;" in custom_css
    assert "justify-content: space-between;" in custom_css
    assert '--md-text-font: "Roboto";' in custom_css
    assert '--md-code-font: "Roboto Mono";' in custom_css
    assert "--md-text-font-family" in custom_css
    assert '"Roboto Mono", SFMono-Regular' in custom_css
    assert "-apple-system" in custom_css
    assert "--psi-brand-width: 20rem;" in custom_css
    assert "--psi-brand-height: 4.5rem;" in custom_css
    assert "--psi-brand-height: 3.25rem;" in custom_css
    assert "--psi-diagram-bg: #ffffff;" in custom_css
    assert "--psi-diagram-ink: #050505;" in custom_css
    assert ".md-header__button.md-logo" in custom_css
    assert "width: 1.2rem;" in custom_css
    assert ".md-nav--primary .md-nav__title .md-nav__button.md-logo" in custom_css
    assert "width: 2.4rem;" in custom_css
    assert ".md-search__form .md-icon svg" in custom_css
    assert "fill: currentcolor;" in custom_css
    assert ".md-nav__button.md-logo" in custom_css
    assert ".psi-footer-wordmark" in custom_css
    assert 'background-image: url("../assets/lllm-logo-text-dark.png");' in custom_css
    assert ".psi-footer-text" in custom_css
    assert "clip-path: inset(50%);" in custom_css
    assert "white-space: nowrap;" in custom_css
    assert ".md-social__link" in custom_css
    assert "height: 2rem;" in custom_css
    assert ".psi-brand img" in custom_css
    assert "max-height: var(--psi-brand-height);" in custom_css
    assert "max-width: min(var(--psi-brand-width), 100%);" in custom_css
    assert 'img[src$="#only-dark"]' in custom_css
    assert ".md-typeset .mermaid svg" in custom_css
    assert "max-width: 100%;" in mermaid_svg_css
    assert "min-width: 0;" in mermaid_svg_css
    assert "width: 100%;" in mermaid_svg_css
    assert "--mermaid-font-family" in custom_css
    assert ".md-typeset .mermaid foreignObject" in custom_css
    assert "line-height: 1.2;" in custom_css
    assert ".md-typeset .mermaid text" in custom_css
    assert ".md-typeset .mermaid .node rect" in custom_css
    assert ".md-typeset .mermaid .edgePath path" in custom_css
    assert ".md-typeset .mermaid marker path" in custom_css
    assert "var(--psi-diagram-ink)" in custom_css
    assert "var fontFamily" in mermaid_js
    assert "Roboto, -apple-system, BlinkMacSystemFont" in mermaid_js
    assert "window.mermaid.startOnLoad = false" in mermaid_js
    assert 'securityLevel: "loose"' in mermaid_js
    assert "flowchart:" in mermaid_js
    assert "htmlLabels: true" in mermaid_js
    assert "themeCSS:" in mermaid_js
    assert "nodeTextColor" in mermaid_js
    assert "useMaxWidth: true" in mermaid_js
    assert "data-mermaid-source" in mermaid_js
    assert "normalizeSource" in mermaid_js
    assert "sourceFor" in mermaid_js
    assert "diagramNodes" in mermaid_js
    assert "renderDiagrams" in mermaid_js
    assert "window.mermaid.run" not in mermaid_js
    assert "falling back to manual rendering" not in mermaid_js
    assert 'document.querySelectorAll(".md-typeset .mermaid, .mermaid")' in mermaid_js
    assert '!node.querySelector("svg")' in mermaid_js
    assert 'node.getAttribute("data-mermaid-rendering") !== "true"' in mermaid_js
    assert "Boolean(source && source.trim())" in mermaid_js
    assert 'node.setAttribute("data-mermaid-source", source)' in mermaid_js
    assert "data-mermaid-error" in mermaid_js
    assert "renderSequence" in mermaid_js
    assert "renderNodeSafely" in mermaid_js
    assert "Mermaid returned an empty SVG." in mermaid_js
    assert "renderAgain" in mermaid_js
    assert "rendering" in mermaid_js
    assert "scheduled" in mermaid_js
    assert "afterFontsReady" in mermaid_js
    assert "data-mermaid-rendering" in mermaid_js
    assert "window.mermaid.render" in mermaid_js
    assert "attempt < maxRetries" in mermaid_js
    assert "requestAnimationFrame" in mermaid_js
    assert "window.document$.subscribe(scheduleRender)" in mermaid_js
    assert 'window.addEventListener("load", scheduleRender)' in mermaid_js
    assert 'window.addEventListener("pageshow", scheduleRender)' in mermaid_js
    assert "renderRunning" not in mermaid_js
    assert "renderScheduled" not in mermaid_js
    assert 'container.getAttribute("data-mermaid-error") === "true"' not in mermaid_js
    assert "assets/logo.svg" in index_html
    assert "assets/lllm-logo-text-dark.png#only-light" in index_html
    assert "assets/lllm-logo-text-white.png#only-dark" in index_html
    assert "psi-footer-wordmark" in index_html
    assert "<div class=\"md-source__repository\">\n    GitHub\n  </div>" in index_html
    assert 'data-md-component="source"' not in index_html
    assert 'src="/assets/lllm-logo-text-dark.png"' not in index_html

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert '<p align="center">' in readme
    assert '<img src="assets/lllm-logo-text-dark.png" alt="LLLM" width="420">' in readme
    assert (site_dir / "CNAME").read_text(encoding="utf-8").strip() == "lllm.one"


def test_docs_nav_keeps_foldable_tutorial_groups():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "- navigation.sections" in config
    assert "- navigation.indexes" in config
    assert "- navigation.expand" not in config
    assert "- navigation.tabs.sticky" not in config
    assert "scheme: slate" not in config
    assert "material/weather-night" not in config
    assert "  - Tutorials:\n      - Protocol Level:" in config
    assert "      - Native Runtime:" in config
    assert "      - Pydantic Runtime:" in config
    assert "          - First Tactic: tutorials/first-tactic.md" in config
    assert "          - Native Core: tutorials/native-core.md" in config
    assert (
        "          - Pydantic AI Compatibility: tutorials/pydantic-ai-compat.md"
        in config
    )


def test_service_api_reference_matches_error_envelope():
    reference = (ROOT / "docs" / "reference" / "service-api.md").read_text(
        encoding="utf-8"
    )

    assert '"detail": {' in reference
    assert '"error": {' in reference
    assert '"type": "SchemaError"' in reference
    assert '"metadata": {}' in reference
    assert "unique method/path pairs" in reference
    assert "reserved LLLM service routes" in reference
    assert "Endpoint paths, names, and tags must avoid whitespace" in reference
    assert "percent escapes" in reference
    assert "network-path prefixes" in reference
    assert '"type": "TacticInputError"' not in reference


def test_protocol_reference_documents_identifier_rules():
    reference = (ROOT / "docs" / "reference" / "protocol.md").read_text(
        encoding="utf-8"
    )

    assert "Tactic names may contain display\nspaces" in reference
    assert "must avoid percent escapes, `.`, `..`, `/`" in reference
    assert "`\\`, and `:`" in reference
    assert "Token-style fields" in reference


def test_psihub_metadata_reference_names_json_schemas():
    reference = (ROOT / "docs" / "reference" / "psihub-metadata.md").read_text(
        encoding="utf-8"
    )

    assert "input and output JSON schemas" in reference
    assert "input and output schema refs" not in reference


def test_composition_guide_documents_mixed_local_config():
    guide = (ROOT / "docs" / "guides" / "composition.md").read_text(
        encoding="utf-8"
    )

    assert "TacticResolver.from_config()" in guide
    assert "shared `.psi/config.toml`" in guide
    assert "non-tactic refs" in guide
    assert "psi://demo/echo/services/api" in guide
    assert "psi://demo/echo/channels/events" in guide
    assert "All binding keys must still be valid" in guide
    assert "known PSI resource sections" in guide
    assert "`schemas`, `services`, `channels`" in guide
    assert "`runs`, `configs`, `docs`" in guide
    assert "unknown resource sections fail validation" in guide
    assert "avoid whitespace" in guide
    assert "do not pad them" in guide
    assert "must be absolute" in guide
    assert "HTTP(S) service URLs" in guide
    assert "must not also declare a `store`" in guide
    assert "`store`, `path`, or `object` target" in guide
    assert "percent escapes" in guide


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


def test_docs_local_asset_references_resolve():
    patterns = [
        re.compile(r'\bsrc="([^"]+)"'),
        re.compile(r"url\([\"']?([^\"')]+)[\"']?\)"),
        re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    ]
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))
    text_paths.extend((ROOT / "docs" / "stylesheets").glob("*.css"))
    skip_prefixes = ("http://", "https://", "mailto:", "data:", "#")

    missing = []
    for path in sorted(text_paths):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            for match in pattern.finditer(text):
                target = match.group(1).strip()
                if (
                    not target
                    or target.startswith(skip_prefixes)
                    or "://" in target
                ):
                    continue
                target = target.split("#", 1)[0].split("?", 1)[0]
                if not target:
                    continue
                if not (path.parent / unquote(target)).resolve().exists():
                    missing.append(
                        f"{path.relative_to(ROOT)} -> {match.group(1)}"
                    )

    assert missing == []
