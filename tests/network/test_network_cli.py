import base64
import json

import pytest

from lllm.network.cli import main


def test_cli_create_list_and_append(tmp_path, capsys):
    store = tmp_path / "store"

    assert (
        main(
            [
                "--store",
                str(store),
                "create-channel",
                "events",
                "--schema",
                "demo.schemas:Event",
                "--description",
                "Event log",
                "--metadata",
                '{"owner": "cli"}',
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["name"] == "events"
    assert created["schema"] == "demo.schemas:Event"
    assert created["description"] == "Event log"
    assert created["metadata"] == {"owner": "cli"}

    assert main(["--store", str(store), "channels"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert listed[0]["name"] == "events"

    assert main(["--store", str(store), "get-channel", "events"]) == 0
    loaded_channel = json.loads(capsys.readouterr().out)
    assert loaded_channel["schema"] == "demo.schemas:Event"
    assert loaded_channel["metadata"] == {"owner": "cli"}

    assert main(["--store", str(store), "append", "events", '{"n": 1}']) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["channel"] == "events"
    assert event["payload"] == {"n": 1}

    assert (
        main(
            [
                "--store",
                str(store),
                "append",
                "events",
                '{"n": 2}',
                "--kind",
                "analysis",
                "--source",
                "cli-test",
                "--schema",
                "demo.schemas:Analysis",
                "--metadata",
                '{"role": "analysis"}',
                "--correlation-id",
                "corr-1",
                "--parent-id",
                event["id"],
            ]
        )
        == 0
    )
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["source"] == "cli-test"
    assert analysis["schema"] == "demo.schemas:Analysis"
    assert analysis["metadata"] == {"role": "analysis"}
    assert analysis["correlation_id"] == "corr-1"
    assert analysis["parent_ids"] == [event["id"]]

    assert main(["--store", str(store), "query-events", "events"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["id"] for item in events] == [event["id"], analysis["id"]]

    assert (
        main(
            [
                "--store",
                str(store),
                "query-events",
                "events",
                "--after-cursor",
                str(event["cursor"]),
                "--kind",
                "analysis",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    filtered = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["id"] for item in filtered] == [analysis["id"]]

    assert main(["--store", str(store), "get-event", event["id"]]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["id"] == event["id"]
    assert loaded["payload"] == {"n": 1}

    assert (
        main(
            [
                "--store",
                str(store),
                "write-artifact",
                "hello",
                "--channel",
                "events",
                "--media-type",
                "text/plain",
                "--metadata",
                '{"name": "greeting"}',
                "--event-id",
                event["id"],
            ]
        )
        == 0
    )
    artifact = json.loads(capsys.readouterr().out)
    assert artifact["channel"] == "events"
    assert artifact["media_type"] == "text/plain"
    assert artifact["metadata"] == {"name": "greeting"}
    assert artifact["event_ids"] == [event["id"]]

    assert main(["--store", str(store), "get-artifact", artifact["id"]]) == 0
    loaded_artifact = json.loads(capsys.readouterr().out)
    assert loaded_artifact["id"] == artifact["id"]
    assert loaded_artifact["size"] == 5

    assert main(["--store", str(store), "read-artifact", artifact["id"]]) == 0
    assert capsys.readouterr().out == "hello\n"

    assert (
        main(
            [
                "--store",
                str(store),
                "read-artifact",
                artifact["id"],
                "--encoding",
                "base64",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == base64.b64encode(b"hello").decode("ascii")

    assert (
        main(
            [
                "--store",
                str(store),
                "put-snapshot",
                "latest",
                '{"status": "ok"}',
                "--channel",
                "events",
                "--schema",
                "demo.schemas:State",
                "--source-event-id",
                event["id"],
                "--metadata",
                '{"role": "latest"}',
            ]
        )
        == 0
    )
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["name"] == "latest"
    assert snapshot["value"] == {"status": "ok"}
    assert snapshot["schema"] == "demo.schemas:State"
    assert snapshot["source_event_id"] == event["id"]
    assert snapshot["metadata"] == {"role": "latest"}

    assert main(["--store", str(store), "get-snapshot", "latest"]) == 0
    loaded_snapshot = json.loads(capsys.readouterr().out)
    assert loaded_snapshot["value"] == {"status": "ok"}

    assert (
        main(
            [
                "--store",
                str(store),
                "create-subscription",
                "events",
                "--id",
                "analysis-worker",
                "--consumer",
                "worker",
                "--batch-size",
                "10",
                "--kind",
                "analysis",
            ]
        )
        == 0
    )
    subscription = json.loads(capsys.readouterr().out)
    assert subscription["id"] == "analysis-worker"
    assert subscription["consumer"] == "worker"
    assert subscription["filters"] == {"kind": "analysis"}

    assert main(["--store", str(store), "pull-subscription", "analysis-worker"]) == 0
    pulled = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["id"] for item in pulled] == [analysis["id"]]

    assert main(["--store", str(store), "get-subscription", "analysis-worker"]) == 0
    loaded_subscription = json.loads(capsys.readouterr().out)
    assert loaded_subscription["cursor"] == analysis["cursor"]

    assert (
        main(
            [
                "--store",
                str(store),
                "create-subscription",
                "events",
                "--id",
                "analysis-worker",
            ]
        )
        == 0
    )
    reused_subscription = json.loads(capsys.readouterr().out)
    assert reused_subscription["cursor"] == analysis["cursor"]

    assert (
        main(
            [
                "--store",
                str(store),
                "append",
                "events",
                '{"n": 3}',
                "--kind",
                "analysis",
            ]
        )
        == 0
    )
    next_analysis = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "--store",
                str(store),
                "pull-subscription",
                "analysis-worker",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    pulled_again = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["id"] for item in pulled_again] == [next_analysis["id"]]


@pytest.mark.parametrize("store", ["   ", " ./store "])
def test_cli_rejects_malformed_store_without_traceback(capsys, store):
    try:
        main(["--store", store, "channels"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected malformed store path to fail")

    output = capsys.readouterr()
    assert output.out == ""
    assert "store root must be a non-empty path string" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "args",
    [
        ["append", "events", "{bad"],
        ["put-snapshot", "latest", "{bad"],
    ],
)
def test_cli_rejects_invalid_json_values_without_traceback(tmp_path, capsys, args):
    store = tmp_path / "store"

    try:
        main(["--store", str(store), *args])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected invalid JSON input to fail")

    output = capsys.readouterr()
    assert output.out == ""
    assert "must be valid JSON" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["create-channel", "events", "--metadata", "{bad"],
            "must be a valid JSON object",
        ),
        (
            ["create-channel", "events", "--metadata", "[]"],
            "must be a JSON object",
        ),
        (
            ["append", "events", '{"n": 1}', "--metadata", "{bad"],
            "must be a valid JSON object",
        ),
        (
            ["write-artifact", "hello", "--metadata", "{bad"],
            "must be a valid JSON object",
        ),
        (
            ["write-artifact", "@@", "--encoding", "base64"],
            "invalid base64 data",
        ),
        (
            ["put-snapshot", "latest", '{"ok": true}', "--metadata", "{bad"],
            "must be a valid JSON object",
        ),
    ],
)
def test_cli_rejects_invalid_option_values_without_traceback(
    tmp_path,
    capsys,
    args,
    expected,
):
    store = tmp_path / "store"

    with pytest.raises(SystemExit) as exc_info:
        main(["--store", str(store), *args])

    assert exc_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert expected in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "args",
    [
        ["create-channel", "bad/name"],
        ["create-channel", "bad%2Fname"],
        ["create-channel", "events", "--form", "not-a-form"],
        ["get-channel", "missing"],
        ["append", "missing", '{"n": 1}'],
        ["append", "bad/name", '{"n": 1}'],
        ["append", "bad%2Fname", '{"n": 1}'],
        ["query-events", "missing"],
        ["query-events", "events", "--limit", "0"],
        ["get-event", "missing"],
        ["create-subscription", "missing"],
        ["create-subscription", "events", "--batch-size", "0"],
        ["get-subscription", "missing"],
        ["pull-subscription", "missing"],
        ["pull-subscription", "bad/name"],
        ["pull-subscription", "bad%2Fname"],
        ["write-artifact", "hello", "--channel", "missing"],
        ["write-artifact", "hello", "--event-id", "missing"],
        ["get-artifact", "missing"],
        ["read-artifact", "missing"],
        ["put-snapshot", "latest", '{"ok": true}', "--channel", "missing"],
        ["put-snapshot", "latest", '{"ok": true}', "--source-event-id", "missing"],
        ["get-snapshot", "missing"],
        ["get-snapshot", "bad/name"],
        ["get-snapshot", "bad%2Fname"],
    ],
)
def test_cli_reports_store_errors_without_traceback(tmp_path, capsys, args):
    store = tmp_path / "store"
    main(["--store", str(store), "create-channel", "events"])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(["--store", str(store), *args])

    assert exc_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "error:" in output.err
    assert "Traceback" not in output.err


def test_cli_reports_binary_artifact_text_decode_error_without_traceback(tmp_path, capsys):
    store = tmp_path / "store"

    assert main(["--store", str(store), "write-artifact", "/w==", "--encoding", "base64"]) == 0
    artifact = json.loads(capsys.readouterr().out)

    with pytest.raises(SystemExit) as exc_info:
        main(["--store", str(store), "read-artifact", artifact["id"]])

    assert exc_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "artifact payload is not valid UTF-8; use --encoding base64" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "--host", ""],
        ["serve", "--host", " 127.0.0.1 "],
        ["serve", "--host", "bad host"],
        ["serve", "--host", "http://127.0.0.1"],
        ["serve", "--port", "0"],
        ["serve", "--port", "70000"],
    ],
)
def test_cli_serve_rejects_malformed_bindings_before_store(tmp_path, capsys, args):
    store = tmp_path / "store"

    try:
        main(["--store", str(store), *args])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected malformed serve binding to fail")

    output = capsys.readouterr()
    assert output.out == ""
    assert "serve " in output.err
    assert not store.exists()
