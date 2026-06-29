(function () {
  var fontFamily =
    "Roboto, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
  var initialized = false;
  var rendering = false;
  var renderAgain = false;
  var scheduled = false;
  var renderSequence = 0;
  var retryDelay = 150;
  var maxRetries = 40;

  function initializeMermaid() {
    if (!window.mermaid) {
      return false;
    }

    if (!initialized) {
      window.mermaid.startOnLoad = false;
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        fontFamily: fontFamily,
        securityLevel: "loose",
        flowchart: {
          htmlLabels: true,
          useMaxWidth: true
        },
        themeCSS:
          ".label, .label text, .label span, .nodeLabel, .edgeLabel, .cluster-label, .cluster-label span { font-family: " +
          fontFamily +
          " !important; }",
        themeVariables: {
          primaryColor: "#ffffff",
          primaryTextColor: "#050505",
          primaryBorderColor: "#050505",
          lineColor: "#050505",
          secondaryColor: "#f7f7f7",
          tertiaryColor: "#ffffff",
          textColor: "#050505",
          fontFamily: fontFamily,
          altFontFamily: fontFamily,
          fontSize: "16px",
          nodeBorder: "#050505",
          nodeTextColor: "#050505",
          mainBkg: "#ffffff",
          clusterBkg: "#ffffff",
          edgeLabelBackground: "#ffffff"
        }
      });
      initialized = true;
    }

    return true;
  }

  function sourceFor(node) {
    var existing = node.getAttribute("data-mermaid-source");
    var code = node.querySelector(
      "code.language-mermaid, code.highlight-mermaid, code"
    );

    return existing || (code ? code.textContent : node.textContent);
  }

  function diagramNodes() {
    return Array.prototype.slice
      .call(document.querySelectorAll(".mermaid"))
      .filter(function (node) {
        return (
          !node.querySelector("svg") &&
          node.getAttribute("data-mermaid-rendering") !== "true" &&
          Boolean(sourceFor(node) && sourceFor(node).trim())
        );
      });
  }

  function renderNode(node, index) {
    var source = sourceFor(node);
    var id = "psi-mermaid-" + Date.now() + "-" + renderSequence + "-" + index;

    renderSequence += 1;
    node.removeAttribute("data-processed");
    node.removeAttribute("data-mermaid-error");
    node.setAttribute("data-mermaid-rendering", "true");
    node.setAttribute("data-mermaid-source", source);
    node.textContent = "";

    if (typeof window.mermaid.render !== "function") {
      return Promise.reject(new Error("No Mermaid render API is available."));
    }

    return Promise.resolve(window.mermaid.render(id, source)).then(function (
      result
    ) {
      var svg = typeof result === "string" ? result : result.svg;

      if (!svg) {
        throw new Error("Mermaid returned an empty SVG.");
      }

      node.innerHTML = svg;
      node.setAttribute("data-processed", "true");

      if (result && typeof result.bindFunctions === "function") {
        result.bindFunctions(node);
      }
    });
  }

  function renderNodeSafely(node, index) {
    return renderNode(node, index)
      .catch(function (error) {
        var source = node.getAttribute("data-mermaid-source") || "";

        node.removeAttribute("data-processed");
        node.setAttribute("data-mermaid-error", "true");
        node.textContent = source;
        console.error("Mermaid render failed", error);
      })
      .then(function () {
        node.removeAttribute("data-mermaid-rendering");
      });
  }

  function renderMermaid(attempt) {
    var nodes;

    if (rendering) {
      renderAgain = true;
      return;
    }

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

    rendering = true;
    Promise.all(nodes.map(renderNodeSafely)).then(
      function () {
        rendering = false;

        if (renderAgain) {
          renderAgain = false;
          scheduleRender();
        }
      },
      function () {
        rendering = false;
      }
    );
  }

  function afterFontsReady(callback) {
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(callback, callback);
    } else {
      callback();
    }
  }

  function scheduleRender() {
    if (scheduled) {
      return;
    }

    scheduled = true;
    window.requestAnimationFrame(function () {
      afterFontsReady(function () {
        scheduled = false;
        renderMermaid(0);
      });
    });
  }

  if (window.document$) {
    window.document$.subscribe(scheduleRender);
  } else {
    document.addEventListener("DOMContentLoaded", scheduleRender);
  }

  window.addEventListener("load", scheduleRender);
  window.addEventListener("pageshow", scheduleRender);
  initializeMermaid();
  scheduleRender();
})();
