"""Deploy the hosting targets in firebase.json with the Firebase Hosting REST API.

The docs workflow logs in to Google Cloud without keys (Workload Identity
Federation) and passes a short-lived access token in FIREBASE_ACCESS_TOKEN;
firebase-tools cannot use such credentials, so this script deploys instead.
It supports what firebase.json uses here: static files, rewrites to files or
Cloud Run services, redirects, cleanUrls, and trailingSlash.

    FIREBASE_ACCESS_TOKEN=... python scripts/deploy_firebase_hosting.py --project PROJECT
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

API = "https://firebasehosting.googleapis.com/v1beta1"
BATCH = 1000  # files per populateFiles request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", default="firebase.json")
    parser.add_argument("--rc", default=".firebaserc")
    parser.add_argument("--only", nargs="*", help="Hosting targets to deploy (default: all)")
    args = parser.parse_args(argv)
    token = os.environ.get("FIREBASE_ACCESS_TOKEN")
    if not token:
        parser.error("set FIREBASE_ACCESS_TOKEN")

    root = Path(args.config).resolve().parent
    hosting = json.loads(Path(args.config).read_text(encoding="utf-8"))["hosting"]
    targets = json.loads(Path(args.rc).read_text(encoding="utf-8"))["targets"][args.project]["hosting"]
    api = _Api(token, args.project)
    for entry in hosting if isinstance(hosting, list) else [hosting]:
        target = entry["target"]
        if args.only and target not in args.only:
            continue
        site = targets[target][0]
        url = api.deploy(site, root / entry["public"], serving_config(entry))
        print(f"released {target} -> {site}: {url}")
    return 0


def serving_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Translate a firebase.json hosting entry into the API's ServingConfig."""

    config: dict[str, Any] = {}
    rewrites = []
    for rule in entry.get("rewrites", []):
        rewrite: dict[str, Any] = {"glob": rule["source"]}
        if "run" in rule:
            rewrite["run"] = {
                "serviceId": rule["run"]["serviceId"],
                "region": rule["run"].get("region", "us-central1"),
            }
        else:
            rewrite["path"] = rule["destination"]
        rewrites.append(rewrite)
    if rewrites:
        config["rewrites"] = rewrites
    redirects = [
        {"glob": rule["source"], "location": rule["destination"], "statusCode": rule.get("type", 301)}
        for rule in entry.get("redirects", [])
    ]
    if redirects:
        config["redirects"] = redirects
    if "cleanUrls" in entry:
        config["cleanUrls"] = bool(entry["cleanUrls"])
    if "trailingSlash" in entry:
        config["trailingSlashBehavior"] = "ADD" if entry["trailingSlash"] else "REMOVE"
    return config


class _Api:
    def __init__(self, token: str, project: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}", "x-goog-user-project": project}

    def call(self, method: str, url: str, body: Any = None, data: bytes | None = None) -> Any:
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"{method} {url}: {exc.code} {exc.read().decode()[:500]}") from exc
        return json.loads(payload) if payload else {}

    def deploy(self, site: str, public: Path, config: dict[str, Any]) -> str:
        version = self.call("POST", f"{API}/sites/{site}/versions", {"config": config})["name"]
        files: dict[str, str] = {}
        blobs: dict[str, bytes] = {}
        for path in sorted(public.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                blob = gzip.compress(path.read_bytes(), mtime=0)
                digest = hashlib.sha256(blob).hexdigest()
                files["/" + path.relative_to(public).as_posix()] = digest
                blobs[digest] = blob
        paths = sorted(files)
        for start in range(0, len(paths), BATCH):
            chunk = {path: files[path] for path in paths[start : start + BATCH]}
            populated = self.call("POST", f"{API}/{version}:populateFiles", {"files": chunk})
            for digest in populated.get("uploadRequiredHashes", []):
                self.call("POST", f"{populated['uploadUrl']}/{digest}", data=blobs[digest])
        self.call("PATCH", f"{API}/{version}?update_mask=status", {"status": "FINALIZED"})
        self.call("POST", f"{API}/sites/{site}/releases?versionName={version}", {})
        return f"https://{site}.web.app"


if __name__ == "__main__":
    raise SystemExit(main())
