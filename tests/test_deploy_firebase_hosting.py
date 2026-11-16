"""The Firebase Hosting deploy script translates firebase.json faithfully."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "deploy_firebase_hosting", ROOT / "scripts" / "deploy_firebase_hosting.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_firebase_json_targets_translate_to_serving_configs():
    script = _script()
    hosting = {entry["target"]: entry for entry in json.loads((ROOT / "firebase.json").read_text())["hosting"]}
    targets = json.loads((ROOT / ".firebaserc").read_text())["targets"]["prosi-501105"]["hosting"]

    # lllm deploys only lllm.one; sssn.one and aaax.one deploy from their repos.
    assert targets == {"docs": ["lllm-one"]}
    assert set(hosting) == {"docs"}
    hub = {"serviceId": "lllm-hub", "region": "us-central1"}
    config = script.serving_config(hosting["docs"])
    assert set(config) == {"rewrites", "redirects"}
    assert config["rewrites"] == [{"glob": "/hub", "run": hub}, {"glob": "/hub/**", "run": hub}]
    assert config["redirects"][:4] == [
        {"glob": "/sssn", "location": "https://sssn.one/", "statusCode": 301},
        {"glob": "/sssn/**", "location": "https://sssn.one/", "statusCode": 301},
        {"glob": "/aaax", "location": "https://aaax.one/", "statusCode": 301},
        {"glob": "/aaax/**", "location": "https://aaax.one/", "statusCode": 301},
    ]
    moved = config["redirects"][4:]  # pages moved in the docs rewrite
    assert moved and all(rule["statusCode"] == 301 and rule["location"].startswith("/") for rule in moved)
    assert script.serving_config({"cleanUrls": True, "trailingSlash": False, "rewrites": [{"source": "**", "destination": "/index.html"}]}) == {
        "rewrites": [{"glob": "**", "path": "/index.html"}],
        "cleanUrls": True,
        "trailingSlashBehavior": "REMOVE",
    }


def test_moved_doc_pages_redirect_to_pages_that_exist():
    docs = ROOT / "docs"
    redirects = json.loads((ROOT / "firebase.json").read_text())["hosting"][0]["redirects"]
    for rule in redirects:
        if rule["destination"].startswith("https://"):
            continue
        path = rule["destination"].split("#")[0].strip("/")
        assert (docs / f"{path or 'index'}.md").exists() or (docs / path / "index.md").exists(), rule
        old = rule["source"].split("{")[0].strip("/")
        assert not (docs / f"{old}.md").exists() and not (docs / old / "index.md").exists(), (
            f"docs/{old} still exists, so its redirect would hide it"
        )


def test_no_docs_page_shadows_the_hub_app():
    # lllm.one/hub is the hub web app, a rewrite to Cloud Run. Firebase serves
    # an exact-match static file before rewrites, so a page there would hide it.
    assert not (ROOT / "docs" / "hub.md").exists()
    assert not (ROOT / "docs" / "hub").exists()

