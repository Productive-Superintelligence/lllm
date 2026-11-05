// Configuration comes from <meta> tags in index.html, so the hub runs as plain
// static files (lllm.one/hub) or behind the upload API (lllm-hub backend).
function metaContent(name) {
  const element = document.querySelector(`meta[name="${name}"]`);
  return element ? element.getAttribute("content") || "" : "";
}

const HUB_BASE = metaContent("lllm-hub-base") || new URL(".", document.baseURI).pathname;
// Empty API base = static catalog only (no live API calls).
const API_BASE = metaContent("lllm-hub-api");
const STATIC_CATALOG_URL = metaContent("lllm-hub-catalog") || `${HUB_BASE}data/catalog.json`;
const SITE_URL = metaContent("lllm-hub-site") || "/";
const DOCS_URL = `${SITE_URL}packages/hub/`;
const HUB_DOCS_URL = `${SITE_URL}packages/hub/`;
const LEGACY_TOKEN =
  localStorage.getItem("psihub_admin_token") || localStorage.getItem("psiweb_admin_token") || "";
let staticCatalogPromise = null;

const state = {
  packages: [],
  selected: null,
  card: "",
  agentCard: "",
  readme: "",
  config: "",
  filters: {
    q: "",
    kind: "",
    resource: "",
    tag: ""
  },
  activeTab: "readme",
  loading: true,
  error: "",
  bundle: null,
  usingStatic: false,
  uploadOpen: false,
  uploadStatus: "",
  mutationBusy: false,
  adminToken: localStorage.getItem("lllm_hub_admin_token") || LEGACY_TOKEN
};

const app = document.querySelector("#app");

function endpoint(path) {
  return `${API_BASE.replace(/\/+$/, "")}${path}`;
}

function hubPath(path = "") {
  return `${HUB_BASE.replace(/\/+$/, "")}/${String(path).replace(/^\/+/, "")}`;
}

async function request(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  if (method !== "GET") return apiRequest(path, options);

  if (!API_BASE) {
    const value = await staticRequest(path);
    state.usingStatic = true;
    return value;
  }

  let apiError = null;
  try {
    const value = await apiRequest(path, options);
    state.usingStatic = false;
    return value;
  } catch (error) {
    apiError = error;
  }

  try {
    const value = await staticRequest(path);
    state.usingStatic = true;
    return value;
  } catch {
    throw apiError;
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(endpoint(path), options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(errorText(text, response.statusText));
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response.text();
}

async function loadStaticCatalog() {
  if (!staticCatalogPromise) {
    staticCatalogPromise = fetch(STATIC_CATALOG_URL)
      .then(response => {
        if (!response.ok) throw new Error("Static catalog unavailable");
        return response.json();
      })
      .then(catalog => {
        state.bundle = catalog.bundle || null;
        return catalog;
      });
  }
  return staticCatalogPromise;
}

async function staticRequest(path) {
  const catalog = await loadStaticCatalog();
  const [resource, queryString = ""] = path.split("?");
  if (resource === "/packages") {
    return filterStaticPackages(catalog.packages || [], new URLSearchParams(queryString));
  }

  const match = resource.match(/^\/packages\/([^/]+)\/([^/]+)(?:\/(.+))?$/);
  if (!match) throw new Error("Static route unavailable");

  const identifier = `${decodeURIComponent(match[1])}/${decodeURIComponent(match[2])}`;
  const detail = catalog.details?.[identifier];
  if (!detail) throw new Error(`Package not found: ${identifier}`);
  const suffix = match[3] || "";
  if (!suffix) return detail;
  if (suffix === "card") return detail.card || "";
  if (suffix === "agent-card") return detail.agent_card || "";
  if (suffix === "readme") return detail.readme || "";
  if (suffix === "config-template") return detail.config_template || "";
  if (suffix === "download") return hubPath(detail.download_url || "");
  throw new Error("Static route unavailable");
}

function filterStaticPackages(packages, params) {
  const q = String(params.get("q") || "").trim().toLowerCase();
  const kind = String(params.get("kind") || "").trim().toLowerCase();
  const tag = String(params.get("tag") || "").trim().toLowerCase();
  const resource = String(params.get("resource") || "").trim().toLowerCase();
  return packages
    .filter(record => !kind || record.kind === kind)
    .filter(record => !tag || (record.tags || []).some(item => item.toLowerCase() === tag))
    .filter(record => !resource || (record.resources || []).some(item => item.kind === resource))
    .filter(record => !q || searchText(record).includes(q))
    .sort((left, right) => left.identifier.localeCompare(right.identifier));
}

function searchText(record) {
  const pieces = [
    record.identifier,
    record.kind,
    record.description,
    record.summary,
    ...(record.tags || [])
  ];
  (record.resources || []).forEach(resource => {
    pieces.push(resource.kind, resource.name, resource.ref, resource.description || "");
  });
  return pieces.join(" ").toLowerCase();
}

async function loadPackages() {
  state.loading = true;
  state.error = "";
  render();
  const params = new URLSearchParams();
  Object.entries(state.filters).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  try {
    state.packages = await request(`/packages?${params.toString()}`);
    if (!state.selected && state.packages.length) {
      await selectPackage(state.packages[0].identifier, { scroll: false });
      return;
    }
    if (state.selected && !state.packages.some(item => item.identifier === state.selected.identifier)) {
      state.selected = null;
    }
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
    render();
  }
}

async function selectPackage(identifier, { scroll = true } = {}) {
  state.error = "";
  const [org, name] = identifier.split("/");
  state.loading = true;
  render();
  try {
    const path = `/packages/${encodeURIComponent(org)}/${encodeURIComponent(name)}`;
    const [selected, card, agentCard, readme, config] = await Promise.all([
      request(path),
      request(`${path}/card`),
      request(`${path}/agent-card`),
      request(`${path}/readme`).catch(() => ""),
      request(`${path}/config-template`)
    ]);
    state.selected = selected;
    state.card = card;
    state.agentCard = agentCard;
    state.readme = readme;
    state.config = config;
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
    render();
    if (scroll) {
      document.querySelector("#repository")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
}

function render() {
  const stats = packageStats(state.packages);
  app.innerHTML = `
    <div class="shell">
      <header class="site-nav">
        <a href="${SITE_URL}" class="home-mark" aria-label="LLLM home">
          <span class="home-psi" aria-hidden="true">LLLM</span>
        </a>
        <nav class="nav-links" aria-label="Primary">
          <a class="is-active" href="${hubPath()}">Hub</a>
          <a href="${SITE_URL}">Docs</a>
          <a href="${DOCS_URL}">Publishing</a>
          <a href="https://github.com/Productive-Superintelligence/lllm">GitHub</a>
        </nav>
      </header>

      ${state.uploadOpen ? renderUploadPanel() : ""}

      <main>
        <section class="hero" aria-labelledby="hero-title">
          <div class="hero-copy">
            <p class="label">LLLM Hub</p>
            <h1 id="hero-title">LLLM package hub.</h1>
            <hr />
            <p class="section-copy">
              Discover, inspect, download, and serve LLLM packages: tactics, agents, prompts, proxies, channels, and services described by one lllm.toml.
            </p>
            <form class="hero-search" data-action="search">
              <input name="q" type="search" placeholder="Search packages, schemas, services..." value="${escapeAttr(state.filters.q)}" />
              <button class="primary" type="submit">Search</button>
            </form>
            <div class="link-row">
              <a class="text-link" href="#repository">Browse packages</a>
              <a class="text-link" href="${HUB_DOCS_URL}">Read docs</a>
              ${state.bundle ? `<a class="text-link" href="${hubPath(state.bundle.url)}" download>Demo bundle</a>` : ""}
              <button class="text-link" data-action="toggle-upload" type="button">${state.uploadOpen ? "Close upload" : "Upload package"}</button>
            </div>
          </div>
          <div class="hero-board" aria-label="LLLM package index">
            <p class="label">Index</p>
            <div class="board-tabs">
              ${categoryTabs(stats).map(tab => `
                <button class="${state.filters.kind === tab.kind ? "is-active" : ""}" data-kind="${escapeAttr(tab.kind)}" type="button">
                  <span>${tab.icon}</span>${escapeHtml(tab.label)}
                </button>
              `).join("")}
            </div>
            <div class="board-list">
              ${previewPackages(state.packages).map(renderBoardPackage).join("") || `<div class="board-empty">No packages indexed</div>`}
            </div>
          </div>
        </section>

        <section class="category-strip" aria-label="Package categories">
          ${categoryCards(stats).map(renderCategoryCard).join("")}
        </section>

        <section class="trending" aria-labelledby="trending-title">
          <div class="section-title">
            <p class="label">Packages</p>
            <h2 id="trending-title">Featured packages.</h2>
          </div>
          <div class="trend-grid">
            ${trendingPackages(state.packages).map(renderTrendCard).join("") || `<div class="empty">No packages</div>`}
          </div>
        </section>

        <section class="workspace" id="repository" aria-label="Package repository browser">
          <aside class="results" aria-label="Package results">
            <form class="filters" data-action="filters">
              <div class="results-meta">
                <strong>${state.packages.length}</strong>
                <span>${state.packages.length === 1 ? "package" : "packages"}</span>
                <span class="source-pill">${state.usingStatic ? "static" : "live"}</span>
              </div>
              <label>
                <span>Kind</span>
                <select name="kind">
                  ${option("", "All kinds", state.filters.kind)}
                  ${option("tactic", "Tactics", state.filters.kind)}
                  ${option("channel", "Channels", state.filters.kind)}
                  ${option("service", "Services", state.filters.kind)}
                  ${option("app", "Apps", state.filters.kind)}
                  ${option("mixed", "Mixed", state.filters.kind)}
                </select>
              </label>
              <label>
                <span>Resource</span>
                <select name="resource">
                  ${option("", "All resources", state.filters.resource)}
                  ${option("tactic", "Tactics", state.filters.resource)}
                  ${option("channel", "Channels", state.filters.resource)}
                  ${option("service", "Services", state.filters.resource)}
                  ${option("schema", "Schemas", state.filters.resource)}
                  ${option("run", "Runs", state.filters.resource)}
                </select>
              </label>
              <label>
                <span>Tag</span>
                <input name="tag" type="search" placeholder="Any tag" value="${escapeAttr(state.filters.tag)}" />
              </label>
              <button class="secondary full" type="submit">Apply filters</button>
            </form>
            ${state.loading ? `<div class="status">Loading</div>` : ""}
            ${state.error ? `<div class="status error">${escapeHtml(state.error)}</div>` : ""}
            <div class="result-list">
              ${state.packages.map(renderPackageRow).join("") || `<div class="empty">No packages</div>`}
            </div>
          </aside>
          <section class="detail" aria-label="Package detail">
            ${state.selected ? renderDetail(state.selected) : `<div class="empty large">Select a package</div>`}
          </section>
        </section>
      </main>
    </div>
  `;
  wireEvents();
}

function renderUploadPanel() {
  return `
    <form class="upload-panel" data-action="upload">
      <label>
        <span>Admin token</span>
        <input name="token" data-token-input type="password" autocomplete="off" value="${escapeAttr(state.adminToken)}" />
      </label>
      <label>
        <span>Package zip</span>
        <input name="file" type="file" accept=".zip,application/zip" />
      </label>
      <button class="primary" type="submit" ${state.mutationBusy ? "disabled" : ""}>
        ${state.mutationBusy ? "Uploading" : "Upload"}
      </button>
      ${state.uploadStatus ? `<output class="upload-status">${escapeHtml(state.uploadStatus)}</output>` : ""}
    </form>
  `;
}

function renderBoardPackage(record) {
  return `
    <button class="board-package" data-package="${escapeAttr(record.identifier)}" type="button">
      <span class="repo-dot ${escapeAttr(record.kind || "package")}"></span>
      <span class="board-main">
        <strong>${escapeHtml(record.identifier)}</strong>
        <small>${escapeHtml(record.summary || record.description || "LLLM package")}</small>
      </span>
      <span>${resourceCount(record)}</span>
    </button>
  `;
}

function renderCategoryCard(card) {
  return `
    <button class="category-card ${state.filters.kind === card.kind ? "is-active" : ""}" data-kind="${escapeAttr(card.kind)}" type="button">
      <span class="category-icon">${card.icon}</span>
      <strong>${escapeHtml(card.label)}</strong>
      <small>${card.count} ${card.count === 1 ? "package" : "packages"}</small>
    </button>
  `;
}

function renderTrendCard(record) {
  const tags = (record.tags || []).slice(0, 2).map(tag => `<span>${escapeHtml(tag)}</span>`).join("");
  return `
    <button class="trend-card" data-package="${escapeAttr(record.identifier)}" type="button">
      <span class="trend-kind">${kindIcon(record.kind)} ${escapeHtml(record.kind || "package")}</span>
      <strong>${escapeHtml(record.identifier)}</strong>
      <p>${escapeHtml(record.summary || record.description || "LLLM package")}</p>
      <span class="trend-meta">
        <span>${resourceCount(record)} resources</span>
        <span>${escapeHtml(record.version || "0.1.0")}</span>
      </span>
      <span class="badges">${tags}</span>
    </button>
  `;
}

function renderPackageRow(record) {
  const selected = state.selected?.identifier === record.identifier;
  const tags = (record.tags || []).slice(0, 3).map(tag => `<span>${escapeHtml(tag)}</span>`).join("");
  return `
    <button class="result ${selected ? "is-selected" : ""}" data-package="${escapeAttr(record.identifier)}" type="button">
      <span class="result-main">
        <span class="repo-dot ${escapeAttr(record.kind || "package")}"></span>
        <span class="result-title">${escapeHtml(record.identifier)}</span>
        <span class="kind-pill">${escapeHtml(record.kind || "package")}</span>
      </span>
      <span class="result-summary">${escapeHtml(record.summary || record.description || "LLLM package")}</span>
      <span class="badges">${tags}</span>
      <span class="result-foot">${resourceCount(record)} resources</span>
    </button>
  `;
}

function renderDetail(record) {
  const resources = Array.isArray(record.resources) ? record.resources : [];
  const tabs = [
    ["readme", "README"],
    ["resources", `Resources ${resources.length}`],
    ["card", "Card"],
    ["config", "Run Config"],
    ["agent", "Agent Card"],
    ["files", "Files"]
  ];
  return `
    <div class="repo-head">
      <div class="repo-title">
        <span class="eyebrow">${kindIcon(record.kind)} ${escapeHtml(record.kind || "package")}</span>
        <h1>${escapeHtml(record.identifier)}</h1>
        <p>${escapeHtml(record.summary || record.description || "")}</p>
      </div>
      <div class="commands">
        ${copyButton(`lllm hub get ${record.identifier}`, "Get")}
        ${copyButton(`lllm hub card ${record.identifier}`, "Card")}
        ${downloadLink(record)}
        <button class="danger" data-action="delete-package" type="button">Delete</button>
      </div>
    </div>
    <div class="meta-row">
      <span>Version ${escapeHtml(record.version || "unknown")}</span>
      <span>Primary ${escapeHtml(record.primary || "none")}</span>
      <span>${resources.length} resources</span>
      ${(record.tags || []).slice(0, 5).map(tag => `<span>${escapeHtml(tag)}</span>`).join("")}
    </div>
    ${renderUsePanel(record)}
    <nav class="tabs" aria-label="Package sections">
      ${tabs.map(([id, label]) => `
        <button class="${state.activeTab === id ? "is-active" : ""}" data-tab="${id}" type="button">
          ${escapeHtml(label)}
        </button>
      `).join("")}
    </nav>
    <div class="panel">
      ${renderActiveTab(record)}
    </div>
  `;
}

function renderUsePanel(record) {
  const commands = usageCommands(record);
  return `
    <section class="use-panel" aria-labelledby="use-package-title">
      <div class="use-heading">
        <div>
          <span class="eyebrow">Use this package</span>
          <h2 id="use-package-title">Run it as a FastAPI service</h2>
        </div>
        ${copyButton(commands.map(item => item.command).join("\n\n"), "Copy all")}
      </div>
      <div class="use-steps">
        ${commands.map((item, index) => `
          <div class="use-step">
            <span>${index + 1}</span>
            <strong>${escapeHtml(item.label)}</strong>
            ${codeBlock(item.command)}
          </div>
        `).join("")}
      </div>
    </section>
  `;
}

function usageCommands(record) {
  const folder = packageFolder(record);
  const archive = `${folder}-${record.version || "package"}.zip`;
  const port = launchPort(record);
  const launchPath = `packages/${folder}`;
  return [
    {
      label: "Install",
      command: 'pip install "lllm-core[native,server]"'
    },
    {
      label: "Download",
      command: [
        `mkdir -p "${launchPath}"`,
        `curl -L "${downloadHref(record)}" -o "${archive}"`,
        `unzip -q -o "${archive}" -d "${launchPath}"`
      ].join("\n")
    },
    {
      label: "Launch",
      command: `lllm setup "${launchPath}" && lllm serve "${launchPath}" --port ${port}`
    },
    {
      label: "Call",
      command: sampleCall(record, port)
    }
  ];
}

function sampleCall(record, port) {
  if (isChannelPrimary(record)) {
    return `curl http://127.0.0.1:${port}/channels`;
  }
  return [
    `curl -X POST http://127.0.0.1:${port}/run \\`,
    "  -H 'content-type: application/json' \\",
    "  -d '{\"input\": {}}'"
  ].join("\n");
}

function packageFolder(record) {
  const name = record.name || String(record.identifier || "package").split("/").pop();
  return String(name || "package").replace(/[^a-zA-Z0-9._-]+/g, "-");
}

function primaryResource(record) {
  const primary = String(record.primary || "");
  const [section, name] = primary.split(".");
  const kindBySection = {
    tactics: "tactic",
    services: "service",
    channels: "channel",
    snapshots: "snapshot",
    runs: "run"
  };
  const kind = kindBySection[section];
  if (!kind || !name || !Array.isArray(record.resources)) return null;
  return record.resources.find(resource => resource.kind === kind && resource.name === name) || null;
}

function isChannelPrimary(record) {
  const primary = String(record.primary || "");
  return primary.startsWith("channels.") || record.kind === "channel";
}

function launchPort(record) {
  const primary = primaryResource(record);
  const service = Array.isArray(record.resources)
    ? record.resources.find(resource => resource.kind === "service")
    : null;
  const metadata = primary?.metadata || service?.metadata || {};
  const port = metadata.port;
  return Number.isInteger(port) && port > 0 && port < 65536 ? port : 8000;
}

function renderActiveTab(record) {
  const resources = Array.isArray(record.resources) ? record.resources : [];
  if (state.activeTab === "resources") {
    return `
      <div class="resource-table">
        <div class="resource-row heading">
          <span>Kind</span><span>Name</span><span>Ref</span><span>Description</span>
        </div>
        ${resources.map(resource => `
          <div class="resource-row">
            <span>${escapeHtml(resource.kind)}</span>
            <span>${escapeHtml(resource.name)}</span>
            <code>${escapeHtml(resource.ref)}</code>
            <span>${escapeHtml(resource.description || "")}</span>
          </div>
        `).join("") || `<div class="table-empty">No resources declared</div>`}
      </div>
    `;
  }
  if (state.activeTab === "config") return codeBlock(state.config || "# no local config template");
  if (state.activeTab === "agent") return codeBlock(state.agentCard || "# no agent card");
  if (state.activeTab === "card") return codeBlock(state.card || "# no package card");
  if (state.activeTab === "files") return renderFiles(record);
  return markdownPreview(state.readme || "README not declared.");
}

function renderFiles(record) {
  const files = Object.keys(record.files || {}).sort();
  if (!files.length) return `<div class="empty">No renderable text files in this package</div>`;
  return `
    <div class="file-list">
      ${files.map(file => `
        <button class="file-row" data-file="${escapeAttr(file)}" type="button">
          <code>${escapeHtml(file)}</code>
          <span>${formatBytes((record.files[file] || "").length)}</span>
        </button>
      `).join("")}
    </div>
  `;
}

function markdownPreview(text) {
  const lines = text.split(/\r?\n/).slice(0, 160);
  let inCode = false;
  return `<article class="markdown">${lines.map(line => {
    if (line.startsWith("```")) {
      inCode = !inCode;
      return inCode ? "<pre><code>" : "</code></pre>";
    }
    if (inCode) return `${escapeHtml(line)}\n`;
    if (line.startsWith("# ")) return `<h2>${escapeHtml(line.slice(2))}</h2>`;
    if (line.startsWith("## ")) return `<h3>${escapeHtml(line.slice(3))}</h3>`;
    if (line.startsWith("### ")) return `<h4>${escapeHtml(line.slice(4))}</h4>`;
    if (line.startsWith("- ")) return `<p class="list-line">${escapeHtml(line)}</p>`;
    if (!line.trim()) return `<br />`;
    return `<p>${escapeHtml(line)}</p>`;
  }).join("")}${inCode ? "</code></pre>" : ""}</article>`;
}

function codeBlock(text) {
  return `<pre><code>${escapeHtml(text)}</code></pre>`;
}

function copyButton(value, label) {
  return `<button class="copy" data-copy="${escapeAttr(value)}" data-label="${escapeAttr(label)}" type="button">${escapeHtml(label)}</button>`;
}

function downloadLink(record) {
  return `<a class="download" href="${escapeAttr(downloadHref(record))}" download>Download</a>`;
}

function downloadHref(record) {
  const [org, name] = record.identifier.split("/");
  const apiHref = endpoint(`/packages/${encodeURIComponent(org)}/${encodeURIComponent(name)}/download`);
  return state.usingStatic && record.download_url ? hubPath(record.download_url) : apiHref;
}

function option(value, label, selected) {
  return `<option value="${escapeAttr(value)}" ${selected === value ? "selected" : ""}>${escapeHtml(label)}</option>`;
}

function wireEvents() {
  document.querySelectorAll("[data-action='search']").forEach(form => {
    form.addEventListener("submit", event => {
      event.preventDefault();
      const data = new FormData(event.currentTarget);
      state.filters.q = String(data.get("q") || "").trim();
      state.selected = null;
      loadPackages();
    });
  });
  document.querySelector("[data-action='filters']")?.addEventListener("submit", event => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    state.filters.kind = String(form.get("kind") || "");
    state.filters.resource = String(form.get("resource") || "");
    state.filters.tag = String(form.get("tag") || "").trim();
    state.selected = null;
    loadPackages();
  });
  document.querySelectorAll("[data-kind]").forEach(button => {
    button.addEventListener("click", () => {
      state.filters.kind = button.dataset.kind || "";
      state.selected = null;
      loadPackages();
    });
  });
  document.querySelector("[data-action='toggle-upload']")?.addEventListener("click", () => {
    state.uploadOpen = !state.uploadOpen;
    state.uploadStatus = "";
    render();
  });
  document.querySelector("[data-action='upload']")?.addEventListener("submit", uploadPackage);
  document.querySelector("[data-token-input]")?.addEventListener("input", event => {
    state.adminToken = String(event.currentTarget.value || "").trim();
    if (state.adminToken) {
      localStorage.setItem("lllm_hub_admin_token", state.adminToken);
      localStorage.removeItem("psiweb_admin_token");
    } else {
      localStorage.removeItem("lllm_hub_admin_token");
      localStorage.removeItem("psihub_admin_token");
      localStorage.removeItem("psiweb_admin_token");
    }
  });
  document.querySelector("[data-action='delete-package']")?.addEventListener("click", deleteSelectedPackage);
  document.querySelectorAll("[data-package]").forEach(button => {
    button.addEventListener("click", () => selectPackage(button.dataset.package));
  });
  document.querySelectorAll("[data-tab]").forEach(button => {
    button.addEventListener("click", () => {
      state.activeTab = button.dataset.tab;
      render();
    });
  });
  document.querySelectorAll("[data-file]").forEach(button => {
    button.addEventListener("click", () => {
      const file = button.dataset.file || "";
      const text = state.selected?.files?.[file] || "";
      navigator.clipboard?.writeText(text);
      button.classList.add("is-copied");
      setTimeout(() => button.classList.remove("is-copied"), 900);
    });
  });
  document.querySelectorAll("[data-copy]").forEach(button => {
    button.addEventListener("click", async () => {
      await navigator.clipboard?.writeText(button.dataset.copy || "");
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = button.dataset.label || "Copy"; }, 1200);
    });
  });
}

async function uploadPackage(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const file = form.get("file");
  const token = String(form.get("token") || "").trim();
  if (!token) {
    state.uploadStatus = "Admin token required";
    render();
    return;
  }
  if (!(file instanceof File) || !file.name) {
    state.uploadStatus = "Choose a package zip";
    render();
    return;
  }

  state.adminToken = token;
  localStorage.setItem("lllm_hub_admin_token", token);
  localStorage.removeItem("psiweb_admin_token");
  state.mutationBusy = true;
  state.uploadStatus = "";
  render();

  const payload = new FormData();
  payload.set("file", file);
  try {
    const detail = await request("/packages/upload", {
      method: "POST",
      headers: { authorization: `Bearer ${state.adminToken}` },
      body: payload
    });
    state.uploadStatus = `Uploaded ${detail.identifier}`;
    state.uploadOpen = false;
    state.selected = null;
    await loadPackages();
    await selectPackage(detail.identifier);
  } catch (error) {
    state.uploadStatus = error.message;
  } finally {
    state.mutationBusy = false;
    render();
  }
}

async function deleteSelectedPackage() {
  if (!state.selected) return;
  if (!state.adminToken) {
    state.uploadOpen = true;
    state.uploadStatus = "Admin token required";
    render();
    return;
  }
  const identifier = state.selected.identifier;
  if (!window.confirm(`Delete ${identifier}?`)) return;
  const [org, name] = identifier.split("/");
  state.mutationBusy = true;
  state.error = "";
  render();
  try {
    await request(`/packages/${encodeURIComponent(org)}/${encodeURIComponent(name)}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${state.adminToken}` }
    });
    state.selected = null;
    await loadPackages();
  } catch (error) {
    state.error = error.message;
  } finally {
    state.mutationBusy = false;
    render();
  }
}

function packageStats(packages) {
  const base = { total: packages.length, tactic: 0, channel: 0, service: 0, app: 0, mixed: 0 };
  packages.forEach(record => {
    const kind = record.kind || "";
    if (Object.prototype.hasOwnProperty.call(base, kind)) base[kind] += 1;
  });
  return base;
}

function categoryTabs(stats) {
  return [
    { kind: "", label: "Main", icon: "ψ", count: stats.total },
    { kind: "tactic", label: "Tactics", icon: "✦", count: stats.tactic },
    { kind: "channel", label: "Channels", icon: "▤", count: stats.channel },
    { kind: "service", label: "Services", icon: "◈", count: stats.service },
    { kind: "app", label: "Apps", icon: "◼", count: stats.app },
    { kind: "mixed", label: "Mixed", icon: "◇", count: stats.mixed }
  ];
}

function categoryCards(stats) {
  return categoryTabs(stats).filter(card => card.kind);
}

function previewPackages(packages) {
  return packages.slice(0, 8);
}

function trendingPackages(packages) {
  return [...packages]
    .sort((left, right) => resourceCount(right) - resourceCount(left) || left.identifier.localeCompare(right.identifier))
    .slice(0, 6);
}

function kindIcon(kind) {
  return {
    tactic: "✦",
    channel: "▤",
    service: "◈",
    app: "◼",
    mixed: "◇"
  }[kind] || "ψ";
}

function errorText(text, fallback) {
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) return parsed.detail.map(item => item.msg || String(item)).join("; ");
  } catch {
    return text;
  }
  return text;
}

function resourceCount(record) {
  return Array.isArray(record.resources) ? record.resources.length : 0;
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  return `${Math.round(value / 102.4) / 10} KB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

loadPackages();
