function renderMermaid() {
  if (!window.mermaid) {
    console.warn("Mermaid runtime was not loaded; diagrams will remain as source blocks.");
    return;
  }

  var nodes = Array.prototype.slice
    .call(document.querySelectorAll("pre.mermaid, div.mermaid"))
    .filter(function (node) {
      return node.getAttribute("data-processed") !== "true";
    });
  if (!nodes.length) {
    return;
  }

  window.mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    themeVariables: {
      primaryColor: "#ffffff",
      primaryTextColor: "#050505",
      primaryBorderColor: "#050505",
      lineColor: "#050505",
      secondaryColor: "#f7f7f7",
      tertiaryColor: "#ffffff",
      fontFamily: "Roboto, sans-serif"
    }
  });

  window.mermaid.run({ nodes: nodes }).catch(function (error) {
    console.error("Mermaid render failed", error);
  });
}

if (window.document$) {
  window.document$.subscribe(renderMermaid);
} else {
  document.addEventListener("DOMContentLoaded", renderMermaid);
}

renderMermaid();
