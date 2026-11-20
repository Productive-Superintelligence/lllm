"""JupyterSession.launch_server finds the URL Notebook 6 and 7 print, and
gives up after its timeout instead of hanging on a silent server."""

import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("jupyter_client", reason="the sandbox needs lllm-core[sandbox]")
pytest.importorskip("nbformat", reason="the sandbox needs lllm-core[sandbox]")

from lllm.runtimes.native.sandbox import jupyter  # noqa: E402

TOKEN = "0123456789abcdef0123456789abcdef0123456789abcdef"
NOTEBOOK_7 = [
    "[I 2026-11-20 10:00:00.000 ServerApp] Jupyter Server 2.14.2 is running at:",
    f"[I 2026-11-20 10:00:00.000 ServerApp] http://127.0.0.1:8899/tree?token={TOKEN}",
    f"[I 2026-11-20 10:00:00.000 ServerApp]     http://127.0.0.1:8899/tree?token={TOKEN}",
]
NOTEBOOK_6 = [
    "[I 10:00:00.000 NotebookApp] Jupyter Notebook 6.5.7 is running at:",
    f"[I 10:00:00.000 NotebookApp] http://127.0.0.1:8888/?token={TOKEN}",
]


@pytest.fixture
def fake_jupyter(monkeypatch, tmp_path):
    """Replace the `jupyter notebook` process with a script that prints the
    given lines to stderr, runs `then`, and stays up without printing."""

    started = []
    real_popen = subprocess.Popen

    def install(lines, *, exit_code=None, then=""):
        script = tmp_path / "fake_jupyter.py"
        tail = f"sys.exit({exit_code})" if exit_code is not None else "time.sleep(60)"
        script.write_text(
            "import sys, time\n"
            f"for line in {lines!r}:\n"
            "    print(line, file=sys.stderr, flush=True)\n"
            f"{then}\n"
            f"{tail}\n"
        )

        def popen(command, **kwargs):
            assert command[:2] == ["jupyter", "notebook"]
            process = real_popen([sys.executable, str(script)], **kwargs)
            started.append(process)
            return process

        monkeypatch.setattr(jupyter.subprocess, "Popen", popen)
        return started

    yield install
    for process in started:
        if process.poll() is None:
            process.kill()
            process.wait()


def session(tmp_path):
    folder = tmp_path / "first"
    folder.mkdir()
    metadata = {
        "project_root": str(tmp_path),
        "proxy": {"activate_proxies": [], "cutoff_date": None, "deploy_mode": False},
    }
    return jupyter.JupyterSession(name="first", dir=str(folder), metadata=metadata)


def launch(sess, **kwargs):
    """Call launch_server on a thread, so a hang fails the test instead."""

    result = {}
    thread = threading.Thread(
        target=lambda: result.update(url=sess.launch_server(**kwargs)), daemon=True
    )
    started = time.monotonic()
    thread.start()
    thread.join(15)
    if thread.is_alive():
        if sess.server_process is not None:
            sess.server_process.kill()
        pytest.fail("launch_server() hung")
    return result["url"], time.monotonic() - started


def test_finds_the_url_notebook_7_prints(fake_jupyter, tmp_path):
    fake_jupyter(NOTEBOOK_7)
    sess = session(tmp_path)

    url, _ = launch(sess)

    assert url == f"http://127.0.0.1:8899/notebooks/first.ipynb?token={TOKEN}"
    assert sess.server_port == 8899
    sess.shutdown_server()
    assert sess.server_process is None


def test_finds_the_url_notebook_6_prints(fake_jupyter, tmp_path):
    fake_jupyter(NOTEBOOK_6)
    sess = session(tmp_path)

    url, _ = launch(sess)

    assert url == f"http://127.0.0.1:8888/notebooks/first.ipynb?token={TOKEN}"
    sess.shutdown_server()


def test_finds_a_url_without_a_token(fake_jupyter, tmp_path):
    fake_jupyter([NOTEBOOK_7[0], "[I 2026-11-20 10:00:00.000 ServerApp] http://127.0.0.1:8899/tree"])
    sess = session(tmp_path)

    url, _ = launch(sess)

    assert url == "http://127.0.0.1:8899/notebooks/first.ipynb"
    sess.shutdown_server()


def test_keeps_reading_the_server_log_after_the_url(fake_jupyter, tmp_path):
    marker = tmp_path / "all_logged"
    # About 2 MB after the URL: far more than a pipe holds, so the server
    # only gets to the marker if launch_server keeps draining its stderr.
    fake_jupyter(
        NOTEBOOK_6,
        then=(
            "for n in range(20000):\n"
            "    print('[I ServerApp] request', n, 'x' * 80, file=sys.stderr)\n"
            f"open({str(marker)!r}, 'w').close()"
        ),
    )
    sess = session(tmp_path)

    url, _ = launch(sess)
    deadline = time.monotonic() + 20
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert url is not None
    assert marker.exists()
    sess.shutdown_server()


def test_gives_up_on_a_silent_server_after_the_timeout(fake_jupyter, tmp_path):
    started = fake_jupyter(NOTEBOOK_7[:1])  # never prints its URL
    sess = session(tmp_path)

    url, seconds = launch(sess, timeout=1)

    assert url is None
    assert seconds < 10
    assert sess.server_process is None
    assert started[0].wait(5) is not None  # the server was stopped


def test_reports_a_server_that_exits_early(fake_jupyter, tmp_path):
    fake_jupyter(["[C 10:00:00.000 ServerApp] Bad config"], exit_code=1)
    sess = session(tmp_path)

    url, seconds = launch(sess, timeout=10)

    assert url is None
    assert seconds < 5
