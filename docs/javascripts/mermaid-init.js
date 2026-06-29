(function () {
  var fontFamily =
    "Roboto, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
  var initialized = false;
  var renderRunning = false;
  var renderAgain = false;
  var renderScheduled = false;
  var renderSequence = 0;
  var observer = null;
  var observedRoot = null;
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
    return true;
  }

  function sourceFor(node, container) {
    var existing = container.getAttribute("data-mermaid-source");
    var code = node.matches("code")
      ? node
      : container.querySelector("code.language-mermaid, code.highlight-mermaid");

    return existing || (code ? code.textContent : container.textContent);
  }

  function isGeneratedMermaidNode(node) {
    var tagName = node.tagName ? node.tagName.toLowerCase() : "";

    return (
      tagName === "svg" ||
      tagName === "foreignobject" ||
      Boolean(node.ownerSVGElement) ||
      Boolean(node.closest("svg, foreignObject"))
    );
  }

  function normalizeNode(node) {
    var container = node.matches("code") ? node.closest("pre") : node;
    var source;

    if (
      isGeneratedMermaidNode(node) ||
      !container ||
      isGeneratedMermaidNode(container) ||
      container.querySelector("svg") ||
      container.getAttribute("data-mermaid-rendering") === "true"
    ) {
      return null;
    }

    source = sourceFor(node, container);
    if (!source || !source.trim()) {
      return null;
    }

    container.classList.add("mermaid");
    container.removeAttribute("data-mermaid-error");
    container.setAttribute("data-mermaid-source", source);
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

  function renderNode(node, index) {
    var source = node.getAttribute("data-mermaid-source") || node.textContent;
    var id = "psi-mermaid-" + Date.now() + "-" + renderSequence + "-" + index;

    renderSequence += 1;
    node.removeAttribute("data-processed");
    node.removeAttribute("data-mermaid-error");
    node.setAttribute("data-mermaid-rendering", "true");
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

    if (renderRunning) {
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

    renderRunning = true;
    Promise.all(nodes.map(renderNodeSafely)).then(function () {
      renderRunning = false;

      if (renderAgain) {
        renderAgain = false;
        scheduleRender();
      }
    }, function () {
      renderRunning = false;
    });
  }

  function afterFontsReady(callback) {
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(callback, callback);
    } else {
      callback();
    }
  }

  function scheduleRender() {
    if (renderScheduled) {
      return;
    }

    observeContent();
    renderScheduled = true;
    window.requestAnimationFrame(function () {
      afterFontsReady(function () {
        renderScheduled = false;
        renderMermaid(0);
      });
    });
  }

  function isMermaidMutation(node) {
    return (
      node.nodeType === 1 &&
      (node.matches(".mermaid, pre code.language-mermaid, pre code.highlight-mermaid") ||
        Boolean(
          node.querySelector(
            ".mermaid, pre code.language-mermaid, pre code.highlight-mermaid"
          )
        ))
    );
  }

  function observeContent() {
    var root;

    if (!window.MutationObserver) {
      return;
    }

    root =
      document.querySelector("[data-md-component='content']") ||
      document.querySelector(".md-content") ||
      document.body;

    if (!root || root === observedRoot) {
      return;
    }

    if (observer) {
      observer.disconnect();
    }

    observer = new MutationObserver(function (mutations) {
      var changed = mutations.some(function (mutation) {
        return Array.prototype.some.call(mutation.addedNodes, isMermaidMutation);
      });

      if (changed) {
        scheduleRender();
      }
    });
    observer.observe(root, { childList: true, subtree: true });
    observedRoot = root;
  }

  function handleDocumentChange() {
    observeContent();
    scheduleRender();
  }

  if (window.document$) {
    window.document$.subscribe(handleDocumentChange);
  } else {
    document.addEventListener("DOMContentLoaded", handleDocumentChange);
  }

  initializeMermaid();
  window.addEventListener("load", handleDocumentChange);
  window.addEventListener("pageshow", handleDocumentChange);
  handleDocumentChange();
})();
