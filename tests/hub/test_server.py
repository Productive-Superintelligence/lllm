import asyncio
import io
import zipfile

import httpx
import pytest
from pydantic import ValidationError

from lllm.hub import LocalHub
from lllm.hub.server import PublishRequest, ValidateRequest, create_app
from test_lifecycle import make_lifecycle_package


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ValidateRequest(path=b"."),
        lambda: PublishRequest(path=b"."),
        lambda: ValidateRequest(path=""),
        lambda: PublishRequest(path="   "),
    ],
)
def test_local_hub_server_request_models_reject_malformed_paths(factory):
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_local_hub_server_publish_request_rejects_non_bool_validate(value):
    with pytest.raises(ValidationError):
        PublishRequest(path=".", validate=value)


def test_local_hub_server_rejects_blank_request_paths(tmp_path):
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            validation = await client.post("/validate", json={"path": "   "})
            publish = await client.post(
                "/publish",
                json={"path": "", "validate": True},
            )
        return validation, publish

    validation, publish = asyncio.run(run())

    assert validation.status_code == 422
    assert publish.status_code == 422
    assert "package path" in validation.text
    assert "package path" in publish.text


def test_local_hub_server_rejects_non_bool_publish_validate(tmp_path):
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/publish",
                json={"path": ".", "validate": "false"},
            )

    response = asyncio.run(run())

    assert response.status_code == 422
    assert "validate" in response.text


def test_local_hub_server_lifecycle(tmp_path):
    package = make_lifecycle_package(tmp_path)
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            validation = await client.post("/validate", json={"path": str(package)})
            publish = await client.post(
                "/publish",
                json={"path": str(package), "validate": True},
            )
            listed = await client.get("/packages")
            metadata = await client.get("/packages/demo/echo")
            card = await client.get("/packages/demo/echo/card")
            agent_card = await client.get("/packages/demo/echo/agent-card")
            config = await client.get("/packages/demo/echo/config-template")
            download = await client.get("/packages/demo/echo/download")
        return health, validation, publish, listed, metadata, card, agent_card, config, download

    health, validation, publish, listed, metadata, card, agent_card, config, download = asyncio.run(run())

    assert health.json() == {"ok": True}
    assert validation.json()["ok"] is True
    assert publish.json()["key"] == "demo/echo@0.1.0"
    assert listed.json()[0]["identifier"] == "demo/echo"
    assert metadata.json()["key"] == "demo/echo@0.1.0"
    assert "lllm://demo/echo/tactics/echo" in card.text
    assert "Agent Card: demo/echo" in agent_card.text
    assert '[refs."lllm://demo/echo/tactics/echo"]' in config.text
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert "lllm.toml" in archive.namelist()


def test_local_hub_server_download_skips_late_secret_files_and_symlinks(tmp_path):
    package = make_lifecycle_package(tmp_path)
    hub = LocalHub(tmp_path / "hub")
    record = hub.publish(package)
    (record.root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (record.root / ".env.example").write_text("TOKEN=\n", encoding="utf-8")
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("TOKEN=outside\n", encoding="utf-8")
    symlink_created = False
    try:
        (record.root / "linked-secret.txt").symlink_to(outside)
        symlink_created = True
    except OSError:
        pass
    app = create_app(hub=hub)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/packages/demo/echo/download")

    response = asyncio.run(run())

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert ".env" not in names
        assert ".env.example" in names
        if symlink_created:
            assert "linked-secret.txt" not in names


def test_local_hub_server_filters_secret_metadata_json_responses(tmp_path):
    package = make_lifecycle_package(tmp_path)
    manifest = package / "lllm.toml"
    text = manifest.read_text(encoding="utf-8").replace(
        '[services.api.metadata]\npolicy_url = "http://policy"',
        '[services.api.metadata]\n'
        'policy_url = "http://policy"\n'
        'api_key = "raw-service-key"\n'
        'apiKey = "raw-camel-service-key"\n'
        'apikey = "raw-compact-service-key"\n'
        'api_key_ref = "credentials/policy"\n'
        'apiKeyRef = "credentials/camel-policy"\n'
        'apikeyref = "credentials/compact-policy"\n'
        'cookie = "raw-service-cookie"\n'
        'sessionCookie = "raw-service-session-cookie"\n'
        'cookie_ref = "credentials/cookie"\n'
        'cookieRef = "credentials/camel-cookie"\n'
        'cookieref = "credentials/compact-cookie"',
    )
    manifest.write_text(
        text
        + """

[config.schema]
type = "object"

[config.schema.properties.password]
type = "string"

[config.schema.properties.accessToken]
type = "string"

[config.defaults]
api_key = "raw-config-key"
accessToken = "raw-camel-config-token"
accesstoken = "raw-compact-config-token"
sessionCookie = "raw-config-cookie"
api_key_ref = "credentials/openai"
apiKeyRef = "credentials/camel-openai"
apikeyref = "credentials/compact-openai"
cookie_ref = "credentials/cookie-default"
cookieRef = "credentials/camel-cookie-default"
cookieref = "credentials/compact-cookie-default"
mode = "safe-mode"

[services.api.metadata.headers]
authorization = "Bearer raw-service-auth"
"set-cookie" = "raw-service-set-cookie"
sessionCookie = "raw-service-header-cookie"
xAuthToken = "raw-camel-service-token"
xauthtoken = "raw-compact-service-token"
x_policy = "safe-policy"
""",
        encoding="utf-8",
    )
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            # Validation rejects these keys; responses must still filter them when skipped.
            publish = await client.post(
                "/publish",
                json={"path": str(package), "validate": False},
            )
            listed = await client.get("/packages")
            metadata = await client.get("/packages/demo/echo")
        return publish, listed, metadata

    publish, listed, metadata = asyncio.run(run())

    for response in (publish, listed, metadata):
        text = response.text
        assert "raw-service-key" not in text
        assert "raw-camel-service-key" not in text
        assert "raw-compact-service-key" not in text
        assert "raw-service-cookie" not in text
        assert "raw-service-session-cookie" not in text
        assert "raw-service-set-cookie" not in text
        assert "raw-service-header-cookie" not in text
        assert "raw-service-auth" not in text
        assert "raw-camel-service-token" not in text
        assert "raw-compact-service-token" not in text
        assert "raw-config-key" not in text
        assert "raw-camel-config-token" not in text
        assert "raw-compact-config-token" not in text
        assert "raw-config-cookie" not in text
        assert "credentials/policy" in text
        assert "credentials/camel-policy" in text
        assert "credentials/compact-policy" in text
        assert "credentials/cookie" in text
        assert "credentials/camel-cookie" in text
        assert "credentials/compact-cookie" in text
        assert "credentials/openai" in text
        assert "credentials/camel-openai" in text
        assert "credentials/compact-openai" in text
        assert "credentials/cookie-default" in text
        assert "credentials/camel-cookie-default" in text
        assert "credentials/compact-cookie-default" in text
        assert "safe-policy" in text
        assert "safe-mode" in text
        assert "password" in text
        assert "accessToken" in text

    config_resource = next(
        resource
        for resource in metadata.json()["resources"]
        if resource["kind"] == "config"
    )
    assert "api_key" not in config_resource["metadata"]["defaults"]
    assert "accessToken" not in config_resource["metadata"]["defaults"]
    assert "accesstoken" not in config_resource["metadata"]["defaults"]
    assert "sessionCookie" not in config_resource["metadata"]["defaults"]
    assert (
        config_resource["metadata"]["defaults"]["apiKeyRef"]
        == "credentials/camel-openai"
    )
    assert (
        config_resource["metadata"]["defaults"]["apikeyref"]
        == "credentials/compact-openai"
    )
    assert (
        config_resource["metadata"]["defaults"]["cookieRef"]
        == "credentials/camel-cookie-default"
    )
    assert (
        config_resource["metadata"]["defaults"]["cookieref"]
        == "credentials/compact-cookie-default"
    )
    assert (
        config_resource["metadata"]["schema"]["properties"]["password"]["type"]
        == "string"
    )
    assert (
        config_resource["metadata"]["schema"]["properties"]["accessToken"]["type"]
        == "string"
    )


def test_local_hub_server_respects_version_query_and_numeric_latest(tmp_path):
    package = make_lifecycle_package(tmp_path)
    manifest = package / "lllm.toml"
    hub = LocalHub(tmp_path / "hub")

    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace('version = "0.1.0"', 'version = "0.2.0"'),
        encoding="utf-8",
    )
    hub.publish(package)

    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace('version = "0.2.0"', 'version = "0.10.0"').replace(
            'summary = "Echo tactic package."',
            'summary = "Newer numeric echo package."',
        ),
        encoding="utf-8",
    )
    hub.publish(package)
    app = create_app(hub=hub)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            latest = await client.get("/packages/demo/echo")
            older = await client.get(
                "/packages/demo/echo",
                params={"version": "0.2.0"},
            )
            older_card = await client.get(
                "/packages/demo/echo/card",
                params={"version": "0.2.0"},
            )
            older_download = await client.get(
                "/packages/demo/echo/download",
                params={"version": "0.2.0"},
            )
            bad_version = await client.get(
                "/packages/demo/echo",
                params={"version": "bad version"},
            )
        return latest, older, older_card, older_download, bad_version

    latest, older, older_card, older_download, bad_version = asyncio.run(run())

    assert latest.json()["version"] == "0.10.0"
    assert older.json()["version"] == "0.2.0"
    assert "Echo tactic package." in older_card.text
    assert "Newer numeric echo package." not in older_card.text
    assert bad_version.status_code == 400
    assert "package version" in bad_version.text
    with zipfile.ZipFile(io.BytesIO(older_download.content)) as archive:
        manifest_text = archive.read("lllm.toml").decode()
    assert 'version = "0.2.0"' in manifest_text


def test_local_hub_server_rejects_invalid_publish(tmp_path):
    package = make_lifecycle_package(tmp_path)
    text = (package / "lllm.toml").read_text(encoding="utf-8")
    (package / "lllm.toml").write_text(
        text.replace('tactic = "echo"', 'tactic = "missing"'),
        encoding="utf-8",
    )
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            publish = await client.post(
                "/publish",
                json={"path": str(package), "validate": True},
            )
            listed = await client.get("/packages")
        return publish, listed

    publish, listed = asyncio.run(run())

    assert publish.status_code == 400
    assert publish.json()["detail"]["ok"] is False
    assert publish.json()["detail"]["issues"][0]["code"] == "service_tactic_missing"
    assert listed.json() == []


def test_local_hub_server_rejects_invalid_package_identifiers(tmp_path):
    app = create_app(hub=LocalHub(tmp_path / "hub"))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return [
                await client.get(path)
                for path in (
                    "/packages/bad:org/echo",
                    "/packages/demo/bad:name",
                    "/packages/bad%20org/echo",
                    "/packages/demo/bad%20name",
                    "/packages/bad%25org/echo",
                    "/packages/demo/bad%25name",
                    "/packages/bad:org/echo/card",
                    "/packages/bad:org/echo/agent-card",
                    "/packages/bad:org/echo/config-template",
                    "/packages/bad:org/echo/download",
                )
            ]

    responses = asyncio.run(run())

    assert {response.status_code for response in responses} == {400}
    assert all("path segment" in response.json()["detail"] for response in responses)
