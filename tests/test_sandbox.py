"""Tests for the platform-native sandbox in treevee.Evaluator.

Linux uses bubblewrap directly; macOS uses zerobox (Seatbelt).
Pure argv builders are tested per-platform; integration tests are
auto-skipped when the relevant backend isn't installed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from treevee import treevee as treevee_mod
from treevee.treevee import (
    Evaluator,
    _build_bwrap_cmd,
    _build_sandbox_cmd,
    _build_zerobox_cmd,
    _find_zerobox,
)


IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
HAS_BWRAP = shutil.which("bwrap") is not None
HAS_ZEROBOX = shutil.which("zerobox") is not None or (
    Path(sys.executable).parent / "zerobox"
).is_file()


@pytest.fixture
def codebase_dir():
    """Integration-test codebase dir, outside /tmp.

    pytest's tmp_path lives under /tmp, but on macOS that would land
    under a profile-allowed read path which can interact poorly with
    zerobox's policy checks. Use ~/.cache/treevee-tests/... instead.
    """
    base = Path.home() / ".cache" / "treevee-tests"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ────────────────────────────────────────────────────────────
# bwrap argv builder (Linux)
# ────────────────────────────────────────────────────────────


def test_build_bwrap_cmd_defaults(tmp_path: Path):
    cmd = _build_bwrap_cmd("echo hi", tmp_path, allow_network=True, tmpdir="/tmp")
    assert cmd[0] == "bwrap"
    # Critical flags from the original treevee setup.
    assert "--ro-bind" in cmd and "/" in cmd
    assert "--bind" in cmd and str(tmp_path) in cmd
    assert "--dev-bind" in cmd  # GPU access — must NOT be --dev or --bind
    assert "--proc" in cmd
    assert "--unshare-all" in cmd
    assert "--die-with-parent" in cmd
    assert "--share-net" in cmd
    assert cmd[-3:] == ["sh", "-c", "echo hi"]


def test_build_bwrap_cmd_no_network(tmp_path: Path):
    cmd = _build_bwrap_cmd("true", tmp_path, allow_network=False, tmpdir="/tmp")
    assert "--unshare-net" in cmd
    assert "--share-net" not in cmd


def test_build_bwrap_cmd_custom_tmpdir(tmp_path: Path):
    custom_tmp = tmp_path / "scratch"
    cmd = _build_bwrap_cmd("true", tmp_path, tmpdir=str(custom_tmp))
    assert custom_tmp.exists()  # builder must create it
    # bwrap binds the host tmpdir over /tmp inside the sandbox.
    assert str(custom_tmp) in cmd
    assert cmd[cmd.index(str(custom_tmp)) + 1] == "/tmp"


# ────────────────────────────────────────────────────────────
# zerobox argv builder (macOS)
# ────────────────────────────────────────────────────────────


def test_build_zerobox_cmd_defaults(tmp_path: Path):
    cmd = _build_zerobox_cmd("echo hi", tmp_path, allow_network=True, tmpdir="/tmp")
    assert Path(cmd[0]).name == "zerobox"
    assert "--profile" in cmd
    assert "--allow-read=/" in cmd
    assert f"--allow-write={tmp_path},/tmp" in cmd
    assert "-C" in cmd and str(tmp_path) in cmd
    assert "--allow-env" in cmd
    assert "--allow-net" in cmd
    assert cmd[-3:] == ["sh", "-c", "echo hi"]
    assert cmd[cmd.index("--") + 1 :] == ["sh", "-c", "echo hi"]


def test_build_zerobox_cmd_no_network(tmp_path: Path):
    cmd = _build_zerobox_cmd("true", tmp_path, allow_network=False, tmpdir="/tmp")
    assert "--allow-net" not in cmd


# ────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────


def test_dispatcher_picks_bwrap_on_linux(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "linux")
    cmd = _build_sandbox_cmd("echo hi", tmp_path, tmpdir="/tmp")
    assert cmd[0] == "bwrap"


def test_dispatcher_picks_zerobox_on_macos(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "darwin")
    cmd = _build_sandbox_cmd("echo hi", tmp_path, tmpdir="/tmp")
    assert Path(cmd[0]).name == "zerobox"


# ────────────────────────────────────────────────────────────
# Platform / availability guards
# ────────────────────────────────────────────────────────────


def test_windows_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError) as ei:
        Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=True)
    msg = str(ei.value)
    assert "--disable-sandbox" in msg
    assert "WSL2" in msg


def test_missing_bwrap_raises_on_linux(monkeypatch, tmp_path: Path):
    if not IS_LINUX:
        pytest.skip("Linux-only check")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError) as ei:
        Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=True)
    assert "bubblewrap" in str(ei.value) or "bwrap" in str(ei.value)


def test_missing_zerobox_raises_on_macos(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(treevee_mod, "_find_zerobox", lambda: None)
    with pytest.raises(RuntimeError) as ei:
        Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=True)
    assert "zerobox" in str(ei.value)


def test_no_sandbox_skips_check(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(treevee_mod, "_find_zerobox", lambda: None)
    Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=False)


def test_macos_emits_experimental_warning(monkeypatch, caplog, tmp_path: Path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(treevee_mod, "_find_zerobox", lambda: "/fake/zerobox")
    with caplog.at_level("WARNING", logger="treevee"):
        Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=True)
    assert any("experimental" in r.message for r in caplog.records)
    assert any(
        "github.com/lambdaloop/treevee/issues" in r.message for r in caplog.records
    )


def test_linux_does_not_emit_warning(monkeypatch, caplog, tmp_path: Path):
    if not IS_LINUX:
        pytest.skip("Linux-only check")
    with caplog.at_level("WARNING", logger="treevee"):
        Evaluator("echo hi", 30, codebase_dir=tmp_path, sandbox=True)
    assert not any("experimental" in r.message for r in caplog.records)


def test_find_zerobox_falls_back_to_sys_executable_bin(monkeypatch, tmp_path: Path):
    # Simulate a fresh venv layout: sys.executable in tmp_path/bin/python,
    # zerobox alongside it. PATH lookup misses, fallback hits.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("")
    (bin_dir / "python").chmod(0o755)
    fake_zb = bin_dir / "zerobox"
    fake_zb.write_text("#!/bin/sh\nexit 0\n")
    fake_zb.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    assert _find_zerobox() == str(fake_zb)


# ────────────────────────────────────────────────────────────
# End-to-end (auto-skips on a host without the right backend)
# ────────────────────────────────────────────────────────────


def _backend_available() -> bool:
    if IS_WIN:
        return False
    if IS_MAC:
        return HAS_ZEROBOX
    return HAS_BWRAP  # Linux + other Unix


needs_backend = pytest.mark.skipif(
    not _backend_available(), reason="no sandbox backend available on this host"
)


@needs_backend
def test_eval_smoke(codebase_dir: Path):
    ev = Evaluator(
        'echo \'{"score": 1.5, "description": "ok"}\'',
        30,
        codebase_dir=codebase_dir,
        sandbox=True,
    )
    result = ev.run()
    assert result.exit_code == 0, result.output
    assert result.score == 1.5
    assert result.description == "ok"
    assert not result.timed_out


@needs_backend
def test_eval_sees_host_dev(codebase_dir: Path):
    # Regression: GPU-using evals need the host's /dev/* nodes.
    # Linux: bwrap --dev-bind /dev /dev. macOS: /dev is naturally visible.
    ev = Evaluator(
        # Real /dev has hundreds of entries on Linux, dozens on macOS.
        # A stub tmpfs would have ~14. Threshold 20 is safe for both.
        'echo "{\\"score\\": $(ls /dev | wc -l)}"',
        30,
        codebase_dir=codebase_dir,
        sandbox=True,
    )
    result = ev.run()
    assert result.exit_code == 0, result.output
    assert result.score is not None and result.score > 20, result.output


@needs_backend
def test_eval_can_read_codebase(codebase_dir: Path):
    # Regression: the eval must be able to execute scripts inside its own
    # codebase (e.g. .pixi/envs/default/bin/python eval.py).
    script = codebase_dir / "hello.sh"
    script.write_text('#!/bin/sh\necho \'{"score": 2.0}\'\n')
    script.chmod(0o755)
    ev = Evaluator("./hello.sh", 30, codebase_dir=codebase_dir, sandbox=True)
    result = ev.run()
    assert result.exit_code == 0, result.output
    assert result.score == 2.0


@needs_backend
def test_eval_writable_inside_codebase(codebase_dir: Path):
    target = codebase_dir / "scratch.txt"
    ev = Evaluator(
        f'echo hi > {target} && echo \'{{"score": 0}}\'',
        30,
        codebase_dir=codebase_dir,
        sandbox=True,
    )
    result = ev.run()
    assert result.exit_code == 0, result.output
    assert target.exists()


@needs_backend
def test_eval_write_escape_blocked(codebase_dir: Path):
    # Try to write to a path outside both codebase_dir and tmpdir.
    forbidden = codebase_dir.parent / "treevee_should_not_exist.txt"
    if forbidden.exists():
        forbidden.unlink()
    ev = Evaluator(
        f'echo nope > {forbidden}; echo \'{{"score": 0}}\'',
        30,
        codebase_dir=codebase_dir,
        sandbox=True,
    )
    ev.run()
    assert not forbidden.exists()


@needs_backend
def test_eval_timeout_returns_quickly(codebase_dir: Path):
    # User-visible behavior: Evaluator.run returns timed_out=True quickly.
    # (Stray inner sleeps may briefly survive; that's the sandbox's
    # process-group reaping policy, not treevee's contract.)
    ev = Evaluator("sleep 60", 1, codebase_dir=codebase_dir, sandbox=True)
    start = time.time()
    result = ev.run()
    elapsed = time.time() - start
    assert result.timed_out is True
    assert elapsed < 10  # 1s timeout + 5s grace, well under 10s


@needs_backend
def test_tmpdir_env_is_set(codebase_dir: Path):
    # On macOS (zerobox) tmpdir is exposed as TMPDIR. On Linux (bwrap)
    # tmpdir is bind-mounted to /tmp. Either way TMPDIR should be the
    # value the user configured.
    ev = Evaluator(
        'echo "{\\"score\\": 0, \\"description\\": \\"$TMPDIR\\"}"',
        30,
        codebase_dir=codebase_dir,
        sandbox=True,
        tmpdir="/tmp",
    )
    result = ev.run()
    assert result.exit_code == 0, result.output
    assert result.description == "/tmp"
