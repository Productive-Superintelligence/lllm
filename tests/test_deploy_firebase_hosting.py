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

    assert set(hosting) == set(targets) == {"docs", "sssn", "aaax"}
    hub = {"serviceId": "lllm-hub", "region": "us-central1"}
    assert script.serving_config(hosting["docs"]) == {
        "rewrites": [{"glob": "/hub", "run": hub}, {"glob": "/hub/**", "run": hub}],
    }
    assert script.serving_config(hosting["sssn"]) == {
        "redirects": [{"glob": "**", "location": "https://lllm.one/sssn/", "statusCode": 301}]
    }
    assert script.serving_config({"cleanUrls": True, "trailingSlash": False, "rewrites": [{"source": "**", "destination": "/index.html"}]}) == {
        "rewrites": [{"glob": "**", "path": "/index.html"}],
        "cleanUrls": True,
        "trailingSlashBehavior": "REMOVE",
    }
