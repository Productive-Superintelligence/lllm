(function () {
  var fontFamily =
    "Roboto, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
  var initialized = false;
  var retryDelay = 150;
  var maxRetries = 40;

  function initializeMermaid() {
    if (!window.mermaid) {
      return false;
    }

    if (initialized) {
      return true;
    }

    window.mermaid.startOnLoad = false;
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      fontFamily: fontFamily,
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
        fontFamily: fontFamily,
        altFontFamily: fontFamily,
        fontSize: "16px",
        nodeBorder: "#050505",
        mainBkg: "#ffffff",
        clusterBkg: "#ffffff",
        edgeLabelBackground: "#ffffff"
      }
    });

    initialized = true;
    return true;
  }

  function sourceFor(node, container) {
    var existing = container.getAttribute("data-mermaid-source");
    var code = node.matches("code")
      ? node
      : container.querySelector("code.language-mermaid, code.highlight-mermaid");

    return existing || (code ? code.textContent : container.textContent);
  }

  function normalizeNode(node) {
    var container = node.matches("code") ? node.closest("pre") : node;
    var source;

    if (!container || container.querySelector("svg")) {
      return null;
    }

    source = sourceFor(node, container);
    if (!source || !source.trim()) {
      return null;
    }

    container.classList.add("mermaid");
    container.removeAttribute("data-mermaid-error");
    container.removeAttribute("data-processed");
    container.setAttribute("data-mermaid-source", source);
    container.textContent = source;
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
      .map(function (node) {
        var container = node.matches("code") ? node.closest("pre") : node;

        if (!container || seen.indexOf(container) !== -1) {
          return null;
        }

        seen.push(container);
        return normalizeNode(node);
      })
      .filter(Boolean);
  }

  function renderWithRun(nodes) {
    if (typeof window.mermaid.run === "function") {
      return window.mermaid.run({ nodes: nodes });
    }

    if (typeof window.mermaid.init === "function") {
      window.mermaid.init(undefined, nodes);
      return Promise.resolve();
    }

    return Promise.reject(new Error("No Mermaid render API is available."));
  }

  function renderNodeFallback(node, index) {
    var source = node.getAttribute("data-mermaid-source") || node.textContent;
    var id = "psi-mermaid-" + Date.now() + "-" + index;

    if (typeof window.mermaid.render !== "function") {
      return Promise.reject(new Error("No Mermaid fallback API is available."));
    }

    return window.mermaid.render(id, source).then(function (result) {
      node.innerHTML = result.svg;
      node.setAttribute("data-processed", "true");

      if (typeof result.bindFunctions === "function") {
        result.bindFunctions(node);
      }
    });
  }

  function renderMermaid(attempt) {
    var nodes;

    if (!initializeMermaid()) {
      if (attempt < maxRetries) {
        window.setTimeout(function () {
          renderMermaid(attempt + 1);
        }, retryDelay);
      } else {
        console.warn("Mermaid runtime was not loaded.");
      }
      return;
    }

    nodes = diagramNodes();
    if (!nodes.length) {
      return;
    }

    renderWithRun(nodes).catch(function (error) {
      Promise.all(nodes.map(renderNodeFallback)).catch(function (fallbackError) {
        nodes.forEach(function (node) {
          node.removeAttribute("data-processed");
          node.setAttribute("data-mermaid-error", "true");
        });
        console.error("Mermaid render failed", error, fallbackError);
      });
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
  scheduleRender();
})();
