import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAV_SECTIONS = ["Get started", "Guide", "Production", "Integrations", "Examples", "API reference"]
DOCS_SWITCH = [
    ("https://lllm.one/", "LLLM"),
    ("https://sssn.one/", "SSSN"),
    ("https://aaax.one/", "AAAX"),
]
MERMAID_DIAGRAMS = ["flowchart", "sequenceDiagram", "stateDiagram-v2"]


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


def test_docs_lead_with_the_native_design():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    package_doc = (ROOT / "lllm" / "__init__.py").read_text(encoding="utf-8")

    for text in (readme, home):
        assert "Low-Level Language Models" in text
        assert "Tactic.quick(" in text
        for idea in ("Tactic is the program", "Agent is the caller", "Prompt is the function", "Dialog is the mental state"):
            assert idea in text
        assert "Analytica" in text and "Apeiron" in text
    assert "Pydantic" in home
    assert "Low-Level Language Models" in package_doc


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
            assert page.evaluate(
                "(kinds) => kinds.filter((kind) => document.body.innerText.includes(kind))",
                MERMAID_DIAGRAMS,
            ) == []
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
                assert metric["source"].lstrip().startswith(tuple(MERMAID_DIAGRAMS))
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
            assert [
                message
                for message in console_errors
                if "Failed to load resource" not in message
            ] == []
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
                  component: element.getAttribute("data-md-component") || "",
                  src: element.getAttribute("src") || "",
                  text: element.textContent.trim().replace(/\\s+/g, " "),
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
                drawerSections: inspect(".psi-drawer-sections"),
                headerNav: inspect(".psi-header-nav"),
                headerNavLinks: [...document.querySelectorAll(".psi-header-nav__link")]
                  .map((element) => element.textContent.trim().replace(/\\s+/g, " ")),
                palette: inspect(".md-header__option[data-md-component='palette']"),
                sidebarText: document
                  .querySelector(".md-sidebar--primary .md-nav--primary > .md-nav__list")
                  ?.innerText.trim().replace(/\\s+/g, " ") || "",
                source: inspect(".md-header__source .md-source"),
                sourceRepository: document
                  .querySelector(".md-header__source .md-source__repository")
                  ?.textContent.trim().replace(/\\s+/g, " ") || "",
                tabs: inspect(".md-tabs"),
                title: inspect(".md-header__title"),
                brandImages,
              };
            }
            """
        )
        page.goto(
            (site_dir / "agents" / "index.html").as_uri(),
            wait_until="domcontentloaded",
        )
        guide_sidebar = page.locator(
            ".md-sidebar--primary .md-nav--primary > .md-nav__list"
        ).inner_text()
        guide_tab = page.locator(".psi-header-nav__link--active").inner_text()
        page.close()
        browser.close()

    assert metrics["header"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["tabs"] is None
    assert metrics["footer"]["backgroundColor"] == "rgb(253, 253, 252)"  # --psi-paper
    assert metrics["header"]["color"] == "rgb(5, 5, 5)"
    assert metrics["footer"]["color"] == "rgb(5, 5, 5)"
    assert metrics["header"]["boxShadow"] == "none"
    assert metrics["title"]["fontWeight"] == "400"
    assert metrics["headerLogo"]["width"] == pytest.approx(24, abs=1)
    assert metrics["headerLogo"]["height"] == pytest.approx(24, abs=1)
    assert metrics["palette"]["width"] == pytest.approx(0, abs=1)
    assert metrics["palette"]["height"] == pytest.approx(0, abs=1)
    assert metrics["drawerSections"]["display"] == "none"
    assert metrics["headerNav"]["display"] == "flex"
    assert metrics["headerNavLinks"] == NAV_SECTIONS
    # The home page is the Introduction, the first page of "Get started"; a
    # page lists only its own section in the sidebar.
    assert "Installation" in metrics["sidebarText"]
    assert "Service API" not in metrics["sidebarText"]
    assert guide_tab == "Guide"
    assert "Core concepts" in guide_sidebar
    assert "Dialogs" in guide_sidebar
    assert "Installation" not in guide_sidebar
    assert "Service API" not in guide_sidebar
    assert metrics["source"]["component"] == "source"
    assert metrics["sourceRepository"].startswith("Productive-Superintelligence/lllm")
    assert metrics["footer"]["height"] == pytest.approx(44, abs=1)
    assert metrics["footerMark"]["width"] == pytest.approx(100, abs=2)
    assert metrics["footerMark"]["height"] == pytest.approx(27, abs=2)
    assert "Newsreader" in metrics["bodyFont"]
    assert "Space Mono" in metrics["codeFont"]

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
                headerNav: inspect(".psi-header-nav"),
                brandImages: images(".psi-brand img"),
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
                    display: style.display,
                    height: rect.height,
                    src: element.getAttribute("src") || "",
                    width: rect.width,
                };
              };
              return {
                logo: inspect(
                  ".md-nav__button.md-logo img, .md-nav__button.md-logo svg"
                ),
                sections: inspect(".psi-drawer-sections"),
                sectionsText: document
                  .querySelector(".psi-drawer-sections")
                  ?.innerText.trim().replace(/\\s+/g, " ") || "",
                title: inspect('.md-nav--primary .md-nav__title[for="__drawer"]'),
              };
            }
            """
        )
        # The home page has no diagrams; How LLLM works draws the agent loop.
        page.goto(
            (site_dir / "concepts" / "index.html").as_uri(),
            wait_until="domcontentloaded",
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
        diagram_metrics = page.evaluate(
            """
            () => ({
              docWidth: document.documentElement.scrollWidth,
              mermaidWidth: document
                .querySelector(".md-typeset .mermaid")
                .getBoundingClientRect().width,
              svgWidth: document
                .querySelector(".md-typeset .mermaid svg")
                .getBoundingClientRect().width,
            })
            """
        )
        page.close()
        browser.close()

    assert metrics["docWidth"] <= metrics["viewportWidth"] + 1
    assert metrics["header"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert metrics["footer"]["backgroundColor"] == "rgb(253, 253, 252)"  # --psi-paper
    assert metrics["header"]["color"] == "rgb(5, 5, 5)"
    assert metrics["footer"]["color"] == "rgb(5, 5, 5)"
    assert metrics["header"]["height"] == pytest.approx(49, abs=1)
    assert metrics["headerLogo"]["src"] == "assets/logo.svg"
    assert metrics["headerLogo"]["width"] == pytest.approx(24, abs=1)
    assert metrics["headerLogo"]["height"] == pytest.approx(24, abs=1)
    assert metrics["headerNav"]["display"] == "none"
    assert metrics["footer"]["height"] <= 72
    assert metrics["footerMark"]["width"] == pytest.approx(100, abs=2)
    assert metrics["footerMark"]["height"] == pytest.approx(27, abs=2)
    assert "Newsreader" in metrics["bodyFont"]
    assert "Space Mono" in metrics["codeFont"]
    assert diagram_metrics["docWidth"] <= metrics["viewportWidth"] + 1
    assert diagram_metrics["mermaidWidth"] <= metrics["viewportWidth"]
    assert diagram_metrics["svgWidth"] > 0
    assert drawer_metrics["title"]["backgroundColor"] == "rgb(255, 255, 255)"
    assert drawer_metrics["title"]["color"] == "rgb(5, 5, 5)"
    assert drawer_metrics["logo"]["src"] == "assets/logo.svg"
    assert drawer_metrics["logo"]["width"] == pytest.approx(48, abs=1)
    assert drawer_metrics["logo"]["height"] == pytest.approx(48, abs=1)
    assert drawer_metrics["sections"]["display"] == "grid"
    for section in NAV_SECTIONS:
        assert section in drawer_metrics["sectionsText"]
    assert "Installation" not in drawer_metrics["sectionsText"]

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
    custom_css = (site_dir / "stylesheets" / "custom.20260702.css").read_text(
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
    assert '--md-text-font: "Newsreader";' in custom_css
    assert '--md-code-font: "Space Mono";' in custom_css
    assert "--md-text-font-family" in custom_css
    assert '"Space Mono", SFMono-Regular' in custom_css
    assert '"Newsreader", Georgia' in custom_css
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
    assert ".psi-header-nav" in custom_css
    assert ".psi-header-nav__link" in custom_css
    assert ".psi-drawer-sections" in custom_css
    assert ".psi-drawer-sections__link" in custom_css
    assert "overflow-x: auto;" in custom_css
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
    assert "Space Mono, SFMono-Regular" in mermaid_js
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
    assert "psi-header-nav" in index_html
    assert "psi-footer-wordmark" in index_html
    assert (
        "<div class=\"md-source__repository\">\n"
        "    Productive-Superintelligence/lllm\n"
        "  </div>"
        in index_html
    )
    assert 'data-md-component="source"' in index_html
    assert 'class="md-tabs"' not in index_html
    assert 'src="/assets/lllm-logo-text-dark.png"' not in index_html

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert '<p align="center">' in readme
    assert '<img src="assets/lllm-logo-text-dark.png" alt="LLLM" width="420">' in readme
    assert (site_dir / "CNAME").read_text(encoding="utf-8").strip() == "lllm.one"


def nav_pages(config: str) -> list[str]:
    return re.findall(r"^\s*-\s+(?:[^:\n]+:\s+)?([\w./-]+\.md)\s*$", config, re.M)


def test_docs_nav_sections_and_no_orphan_pages():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "- navigation.sections" in config
    assert "- navigation.indexes" in config
    assert "- navigation.tabs" not in config
    assert "scheme: slate" not in config
    # The header renders the top-level sections as tabs. The logo leads home,
    # to the Introduction at the start of "Get started"; there is no Home tab.
    nav = config.split("\nnav:\n", 1)[1].split("\n\n", 1)[0]
    assert re.findall(r"^  - ([^:\n]+):\s*$", nav, re.M) == NAV_SECTIONS
    assert "homepage: https://lllm.one/" in config
    pages = nav_pages(config)
    docs = ROOT / "docs"
    assert [page for page in pages if not (docs / page).is_file()] == []
    assert pages[:3] == ["index.md", "install.md", "concepts.md"]
    all_pages = {
        str(path.relative_to(docs))
        for path in docs.rglob("*.md")
        if "overrides" not in path.parts
    }
    assert sorted(all_pages - set(pages)) == []
    assert not (docs / "sssn").exists()
    assert not (docs / "aaax").exists()


def test_docs_header_has_section_tabs_and_switches_between_the_three_sites(tmp_path):
    site_dir = build_docs(tmp_path)
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    switch = config.split("  docs_switch:\n", 1)[1].split("\n  social:", 1)[0]

    assert re.findall(r"title: (\w+)\n\s+url: (\S+)", switch) == [
        (title, url) for url, title in DOCS_SWITCH
    ]
    for page in ("index.html", "agents/index.html", "api/cli/index.html"):
        html = (site_dir / page).read_text(encoding="utf-8")
        header = html.split('class="psi-header-nav"', 1)[1].split("</div>", 1)[0]
        tabs = re.findall(r'class="psi-header-nav__link[^"]*">\s*(.*?)\s*</a>', header, re.S)
        menu = html.split('class="psi-doc-switch__menu"', 1)[1].split("</div>", 1)[0]
        docs = re.findall(r'<a href="([^"]+)" class="psi-doc-switch__link[^"]*">\s*(\w+)', menu)
        drawer = html.split('class="psi-drawer-sections"', 1)[1].split("</div>", 1)[0]
        drawer_links = re.findall(r'class="psi-drawer-sections__link[^"]*">\s*(.*?)\s*</a>', drawer, re.S)

        assert tabs == NAV_SECTIONS, page
        assert docs == DOCS_SWITCH, page
        assert drawer_links == NAV_SECTIONS + [title for _url, title in DOCS_SWITCH], page
        assert 'href="https://lllm.one/" title="LLLM" class="md-header__button md-logo"' in html


def test_docs_use_sssn_and_aaax_instead_of_the_old_module_paths():
    text_paths = list((ROOT / "docs").rglob("*.md"))
    text_paths.extend((ROOT / "examples").rglob("*.py"))
    text_paths.extend((ROOT / "examples").rglob("*.md"))

    old_paths = re.compile(r"(?<![\w./])lllm[./](?:network|shell)\b")  # not .lllm/network
    for path in text_paths:
        match = old_paths.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path.relative_to(ROOT)} mentions {match.group(0)}"


GUIDE_PAGES = [
    "agents",
    "prompts",
    "tools",
    "dialogs",
    "tactics",
    "typed-io",
    "configuration",
    "streaming",
    "models",
    "inference-hints",
    "proxies",
    "sandbox",
    "serving",
    "testing",
    "observability",
    "packages",
    "publishing",
    "pydantic-ai",
    "runtimes",
]


def test_guide_covers_core_surfaces():
    docs = ROOT / "docs"
    for page in GUIDE_PAGES:
        assert (docs / f"{page}.md").is_file(), page
    text = "\n".join((docs / f"{page}.md").read_text(encoding="utf-8") for page in GUIDE_PAGES)
    for term in (
        "AgentCallSession",
        "TacticCallSession",
        "agent_group",
        "agent_tools",
        "max_rate_limit_retries",
        "CALL_API",
        "InferenceHints",
        "x-dynamo-session-id",
        "tactic_as_function",
        "return_session=True",
        "lllm serve",
        "lllm.testing",
    ):
        assert term in text, term


def test_service_api_reference_matches_error_envelope():
    reference = (ROOT / "docs" / "api" / "service-api.md").read_text(
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
    assert "dot segments" in reference
    assert "semicolon path params" in reference
    assert '"type": "TacticInputError"' not in reference


def test_hub_metadata_reference_names_json_schemas():
    reference = (ROOT / "docs" / "api" / "hub-metadata.md").read_text(
        encoding="utf-8"
    )

    assert "JSON schemas of the typed input and output" in reference
    assert "Package refs use `lllm://org/package/tactics/name`" in reference
    assert "Service refs use either" in reference


def test_refs_page_documents_local_config_and_resolver():
    refs = (ROOT / "docs" / "packages.md").read_text(encoding="utf-8")

    assert "lllm://" in refs
    assert ".lllm/config.toml" in refs
    assert "Resolver" in refs
    assert "url =" in refs


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
        assert "psi://" not in text, path
        assert "psi.toml" not in text, path
        assert "PsiHub" not in text, path
        assert "prosi-psi-cli" not in text, path


def fences(text: str) -> list[tuple[str, str]]:
    """(info string, body) of every top-level fenced code block."""

    return re.findall(r"^```([^\n]*)\n(.*?)^```", text, re.M | re.S)


def test_guide_pages_show_a_runnable_example_with_its_output():
    # The docs teach by example: every guide page has a Python example that
    # tests/test_docs_examples.py runs, with the printed output it checks.
    for page in ["index", "concepts", *GUIDE_PAGES]:
        text = (ROOT / "docs" / f"{page}.md").read_text(encoding="utf-8")
        checked = [
            body
            for info, body in fences(text)
            if info.startswith("python")
            and 'test="skip"' not in info
            and ("#>" in body or '\n"""' in body)
        ]
        assert checked, f"docs/{page}.md has no runnable example with checked output"


CODE_COLUMN = 76  # characters of the code font that fit the content column


def test_docs_code_fits_the_code_column():
    # Longer lines are cut off behind a horizontal scrollbar. Printed output
    # (after `#>`, or in a `"""` block of its own) is exempt: it is what the
    # program prints.
    long_lines = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^([ \t]*)```([^\n]*)\n(.*?)^\1```", text, re.M | re.S):
            if match.group(2).strip().startswith("mermaid"):
                continue
            quote = None
            for line in match.group(3).splitlines():
                stripped = line.strip()
                if quote == "output":
                    quote = None if stripped == '"""' else quote
                    continue
                if quote is None and stripped == '"""':
                    quote = "output"
                    continue
                if quote == "doc" and stripped.endswith('"""'):
                    quote = None
                elif quote is None and stripped.startswith('"""') and not (
                    len(stripped) >= 6 and stripped.endswith('"""')
                ):
                    quote = "doc"
                if stripped.startswith("#>"):
                    continue
                if len(line) - len(match.group(1)) > CODE_COLUMN:
                    long_lines.append(f"{path.relative_to(ROOT)}: {stripped[:70]}")
    assert long_lines == []


def prose(path: Path) -> str:
    """File text with fenced code blocks removed (code is not links)."""

    text = path.read_text(encoding="utf-8")
    if path.suffix != ".md":
        return text
    return re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.M | re.S)


def test_readme_and_docs_local_links_resolve():
    pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))
    skip_prefixes = ("http://", "https://", "mailto:", "#")

    missing = []
    for path in sorted(text_paths):
        for match in pattern.finditer(prose(path)):
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
        text = prose(path)
        for pattern in patterns:
            if pattern.pattern.startswith("url") and path.suffix != ".css":
                continue  # CSS url(...) only applies to stylesheets
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
