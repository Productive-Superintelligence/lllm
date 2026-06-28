(function () {
  var initialized = false;

  function initializeMermaid() {
    if (initialized || !window.mermaid) {
      return;
    }

    window.mermaid.startOnLoad = false;
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      fontFamily:
        "Roboto, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif",
      securityLevel: "strict",
      flowchart: {
        htmlLabels: false,
        useMaxWidth: true
      },
      themeVariables: {
        primaryColor: "#ffffff",
        primaryTextColor: "#050505",
        primaryBorderColor: "#050505",
        lineColor: "#050505",
        secondaryColor: "#f7f7f7",
        tertiaryColor: "#ffffff",
        fontFamily:
          "Roboto, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif",
        altFontFamily:
          "Roboto, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif",
        fontSize: "16px",
        nodeBorder: "#050505",
        mainBkg: "#ffffff",
        clusterBkg: "#ffffff",
        edgeLabelBackground: "#ffffff"
      }
    });
    initialized = true;
  }

  function normalizeDiagramNode(node) {
    var container = node;
    var source = node;
    var sourceText = node.getAttribute("data-mermaid-source");

    if (node.matches("code")) {
      source = node;
      container = node.closest("pre");
      if (!container) {
        return null;
      }
      container.classList.add("mermaid");
    } else {
      source =
        node.matches("pre") || node.matches("div")
          ? node.querySelector("code.language-mermaid, code.highlight-mermaid")
          : null;
    }

    if (!sourceText && source) {
      sourceText = source.textContent;
    }

    if (!sourceText) {
      sourceText = container.textContent;
    }

    if (sourceText) {
      container.setAttribute("data-mermaid-source", sourceText);
      container.textContent = sourceText;
    }

    return container;
  }

  function diagramNodes() {
    var seen = [];
    var nodes = Array.prototype.slice.call(
      document.querySelectorAll(
        ".mermaid, pre code.language-mermaid, pre code.highlight-mermaid"
      )
    );

    return nodes
      .filter(function (node) {
        var container = node.matches("code") ? node.closest("pre") : node;
        if (!container || seen.indexOf(container) !== -1) {
          return false;
        }
        seen.push(container);
        return (
          container.getAttribute("data-processed") !== "true" &&
          container.getAttribute("data-mermaid-error") !== "true"
        );
      })
      .map(normalizeDiagramNode);
  }

  function renderMermaid(attempt) {
    if (!window.mermaid) {
      if (attempt < 30) {
        window.setTimeout(function () {
          renderMermaid(attempt + 1);
        }, 200);
      } else {
        console.warn(
          "Mermaid runtime was not loaded; diagrams will remain as source blocks."
        );
      }
      return;
    }

    initializeMermaid();

    var nodes = diagramNodes();
    if (!nodes.length) {
      return;
    }

    window.mermaid.run({ nodes: nodes }).catch(function (error) {
      nodes.forEach(function (node) {
        node.removeAttribute("data-processed");
        node.setAttribute("data-mermaid-error", "true");
      });
      console.error("Mermaid render failed", error);
    });
  }

  function scheduleRender() {
    window.requestAnimationFrame(function () {
      renderMermaid(0);
    });
  }

  if (window.document$) {
    window.document$.subscribe(scheduleRender);
  } else {
    document.addEventListener("DOMContentLoaded", scheduleRender);
  }

  window.addEventListener("load", scheduleRender);

  initializeMermaid();
  scheduleRender();
})();
