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
      fontFamily: "Roboto, Helvetica Neue, Arial, sans-serif",
      securityLevel: "strict",
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true
      },
      themeVariables: {
        primaryColor: "#ffffff",
        primaryTextColor: "#050505",
        primaryBorderColor: "#050505",
        lineColor: "#050505",
        secondaryColor: "#f7f7f7",
        tertiaryColor: "#ffffff",
        fontFamily: "Roboto, Helvetica Neue, Arial, sans-serif",
        altFontFamily: "Roboto, Helvetica Neue, Arial, sans-serif"
      }
    });
    initialized = true;
  }

  function normalizeDiagramNode(node) {
    var container = node;
    var source = node;

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

    if (source) {
      container.textContent = source.textContent;
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
      .map(normalizeDiagramNode)
      .filter(function (node) {
        if (!node || seen.indexOf(node) !== -1) {
          return false;
        }
        seen.push(node);
        return (
          node.getAttribute("data-processed") !== "true" &&
          node.getAttribute("data-mermaid-error") !== "true"
        );
      });
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
