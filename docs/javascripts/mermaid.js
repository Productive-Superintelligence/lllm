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

  function diagramNodes() {
    return Array.prototype.slice
      .call(document.querySelectorAll("pre.mermaid, div.mermaid"))
      .filter(function (node) {
        return node.getAttribute("data-processed") !== "true";
      });
  }

  function renderMermaid(attempt) {
    if (!window.mermaid) {
      if (attempt < 4) {
        window.setTimeout(function () {
          renderMermaid(attempt + 1);
        }, 100);
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

  initializeMermaid();
  scheduleRender();
})();
