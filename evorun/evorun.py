"""evorun.py — LLM-driven iterative codebase optimization.

A standalone Monte Carlo Tree Search (MCTS) optimizer that iteratively
improves code using LLM feedback. Designed as a lightweight alternative
to the full MLEvolve system, with a single-process execution model.

See EVORUN_DEVELOPER.md for internal architecture, data flow,
state file format, extensibility guide, and planned MLEvolve features.
See EVORUN_GUIDE.md for usage, examples, and troubleshooting.

Features:
    - Evaluates current codebase, queries LLM for improvements, applies changes
    - Evaluates resulting code, measures improvement, repeats
    - Full Monte Carlo Tree Search (MCTS) for branching exploration
    - Auto-restore best checkpoint on restart or Ctrl+C
    - Snapshot-based node tracking for tree search
    - Fake-run mode for rapid prototyping without real eval/LLM calls

Stopping criteria (any condition stops the run):
    1.  Maximum iterations exceeded
    2.  Wall-clock time exceeds --time-limit (if specified)
    3.  All branches stagnant (per-branch patience > N consecutive no-improvement)
    4.  Too many consecutive eval timeouts

Usage:
    python evorun.py ./codebase \\
        --max-iters 20 --patience 5 --reset

LLM configuration (model, API key, base URL) is read from ~/.config/evorun/evorun.toml and the --model CLI flag.
For fake-run (no eval, no LLM, just random scores and tree exploration):
    python evorun.py ./codebase \\
        --model none --fake-run --reset --max-iters 10
"""

import argparse
import concurrent.futures
from datetime import datetime
import math
import threading

import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import tomllib

from evorun.tree_search import TreeSearch
from evorun.utils.metric import MetricValue

_run_logger = logging.getLogger("evorun")

# ────────────────────────────────────────────────────────────
# Configuration constants
# ────────────────────────────────────────────────────────────

# Eval output truncation.
MAX_OUTPUT_LINES = 50
MAX_OUTPUT_LINE_LEN = 200

# Speed warning thresholds (fraction of eval_timeout).
SPEED_CRITICAL_THRESHOLD = 0.9
SPEED_WARNING_THRESHOLD = 0.7

# Stagnation / no-change detection.
MAX_CONSECUTIVE_NO_CHANGES = 3

# Response truncation limits.
MAX_LLM_RESPONSE_SUMMARY = 300
MAX_EDIT_SUMMARY_CHARS = 100
MAX_PREV_EVAL_CHARS = 3000

# Fake-run score variation range.
FAKE_SCORE_DELTA_RANGE = 0.3

# ────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Result of running the evaluation command.

    Attributes:
        score: Parsed score (float) or None if parse failed.
        description: Text describing the evaluation result.
        timed_out: True if eval exceeded the timeout.
        exec_time: Actual seconds elapsed running the command.
        exit_code: Return code from subprocess.
        output: Raw stdout + stderr lines.
        had_error: True if a generic exception occurred (not timed out).
    """
    score: float | None = None
    description: str = ""
    timed_out: bool = False
    exec_time: float = 0.0
    exit_code: int | None = None
    output: list[str] = field(default_factory=list)
    had_error: bool = False


@dataclass
class HistoryEntry:
    """Record of a single iteration (used for history and state persistence).

    Attributes:
        iter: Iteration number.
        score: Eval score for this iteration.
        timed_out: Whether the eval timed out.
        exec_time: Actual execution time.
        datetime: ISO-8601 timestamp of when the iteration completed.
        files_modified: List of file paths that were modified.
        files_added: List of file paths that were added.
        files_deleted: List of file paths that were deleted.
        llm_response: Full LLM response text (stored in state for resume).
        edit_summary: One-line LLM-summarised description of code changes.
        diff_text: Unified diff text sent to the summariser (for debugging).
    """
    iter: int
    score: float | None
    timed_out: bool
    exec_time: float
    datetime: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_added: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    llm_response: str = ""
    edit_summary: str = ""
    diff_text: str = ""


# ────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────


class Evaluator:
    """Runs the evaluation command and parses its JSON output.

    The evaluator handles the execution of the eval command with proper
    timeout management and output parsing.  It supports both JSON-based
    output (preferred) and regex fallback for non-structured output.

    Args:
        eval_cmd: Shell command to execute (e.g., "python eval.py").
        eval_timeout: Maximum seconds to wait for the command to complete.
        codebase_dir: Working directory to run from.

    Output Format:
        The eval command should print JSON to stdout with keys:
            - "score":  numeric value (float)
            - "description": optional text describing the result

    Example:
        evaluator = Evaluator("python evaluate.py", 300)
        result = evaluator.run()
        if result.score is not None:
            print(f"Score: {result.score}")
    """

    def __init__(
        self,
        eval_cmd: str,
        eval_timeout: int,
        codebase_dir: Path = Path("."),
    ):
        """Initialize the evaluator.

        Args:
            eval_cmd: Shell command to execute (e.g., "pixi run python eval.py").
            eval_timeout: Maximum execution time in seconds.
            codebase_dir: Working directory to run from (for pixi env resolution).
        """
        self.eval_cmd = eval_cmd
        self.eval_timeout = eval_timeout
        self.codebase_dir = codebase_dir

    def run(self) -> EvalResult:
        """Execute the evaluation command and parse its output.

        This method runs the eval command from the codebase directory
        with its own pixi environment.  If the command times out or fails,
        appropriate timeout/error flags are set.

        Returns:
            EvalResult with score, description, timeout status, and metadata.
        """
        start = time.time()
        proc = None
        try:
            # start_new_session=True puts the process in its own process
            # group so that kill() terminates not just the child shell but
            # also any subprocesses it spawns (e.g. pixi/env processes).
            proc = subprocess.Popen(
                self.eval_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                cwd=str(self.codebase_dir),
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.eval_timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process group so child processes are also
                # terminated (proc.kill() alone only kills the shell).
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                elapsed = time.time() - start
                return EvalResult(
                    timed_out=True, exec_time=elapsed, exit_code=-1,
                    output=_split_lines(stdout) + _split_lines(stderr),
                )
            elapsed = time.time() - start
            exit_code = proc.returncode
            output = _split_lines(stdout) + _split_lines(stderr)
            score, description = self._parse_output(stdout)
            return EvalResult(
                score=score, description=description, timed_out=False,
                exec_time=elapsed, exit_code=exit_code, output=output,
            )
        except Exception as e:
            # Clean up any still-running subprocess.
            if proc and proc.returncode is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
            elapsed = time.time() - start
            _run_logger.error(f"Evaluation command failed: {e}")
            return EvalResult(
                score=None, description=f"Command error: {e}",
                timed_out=False, exec_time=elapsed, exit_code=None,
                output=[str(e)], had_error=True,
            )

    @staticmethod
    def _parse_output(stdout: str) -> tuple[float | None, str]:
        """Extract score and description from stdout.

        Always reads the last non-empty line of output and tries to parse it
        as JSON first. Falls back to regex on that same line.

        Args:
            stdout: Raw stdout string from the eval command.

        Returns:
            Tuple of (score, description).
        """
        lines = stdout.strip().splitlines()
        if not lines:
            return None, ""

        last_line = lines[-1].strip()
        if not last_line:
            return None, ""

        # Try JSON on the last line.
        try:
            data = json.loads(last_line)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                if math.isnan(score):
                    return None, data.get("description", "")
                return score, data.get("description", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback — require word boundary so "score" isn't part of a
        # larger token (e.g., "no_score" or "autoscore").
        pattern = re.compile(r"(?:^|(?<=\s))score[:=]\s*(\d+(?:\.\d+)?)")
        m = pattern.search(last_line)
        if m:
            return float(m.group(1)), ""
        return None, ""


# ────────────────────────────────────────────────────────────
# Codebase manager
# ────────────────────────────────────────────────────────────


class CodebaseManager:
    """Handles backup and restoration of the experiment directory.

    Backs up experiment/, TASK.md, eval.py, and pixi.toml by explicitly
    copying each to .evorun_backup. Restores everything on demand.
    Snapshots additionally handle checkpointing at best scores.

    Directories Created:
        .evorun_backup/        — experiment/ + TASK.md + eval.py + pixi.toml
        .evorun_snapshots/     — per-node snapshots at best scores

    Args:
        codebase_dir: Path to the directory containing the codebase experiment.
        task_file: Name of the task specification file (default "TASK.md").
    """

    def __init__(self, codebase_dir: str, task_file: str = "TASK.md"):
        """Initialize CodebaseManager.

        Args:
            codebase_dir: Path to the directory containing the codebase experiment.
            task_file: Name of the task spec file to load for LLM context.
        """
        self.codebase_dir = Path(codebase_dir).resolve()
        self.backup_dir = self.codebase_dir / ".evorun_backup"
        self._task_file: Path | None = None
        self._task_content: str | None = None
        self._task_filename = task_file
        self._load_task_file()

    def _load_task_file(self) -> None:
        """Find and read the task specification file from the codebase root."""
        self._task_file = self.codebase_dir / self._task_filename
        if self._task_file.exists():
            try:
                self._task_content = self._task_file.read_text(encoding="utf-8")
                _run_logger.info(f"Loaded task file: {self._task_filename}")
            except OSError:
                self._task_content = None
                self._task_file = None

    def setup(self, resuming: bool = False) -> None:
        """Backup experiment/, TASK.md, eval.py, and pixi.toml.

        Args:
            resuming: If True, skip creating a new backup (resume mode).
        """
        if resuming:
            _run_logger.info("Resuming existing run — skipping backup creation.")
            return
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 1) experiment/ subfolder — all files, recursively
        exp_src = self.codebase_dir / "experiment"
        if exp_src.is_dir():
            for p in exp_src.rglob("*"):
                if p.is_file() and self._should_include_file(p):
                    backup_path = self.backup_dir / p.relative_to(self.codebase_dir)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, backup_path)
            _run_logger.info(f"Backed up experiment/ ({exp_src})")

        # 2) Root-level files — explicit
        for fname in ("TASK.md", "eval.py", "pixi.toml"):
            src = self.codebase_dir / fname
            if src.exists():
                dst = self.backup_dir / fname
                shutil.copy2(src, dst)
                _run_logger.info(f"Backed up {fname}")

    def get_experiment_files(self) -> list[str]:
        """Return list of relative paths for all files in experiment/."""
        exp_src = self.codebase_dir / "experiment"
        if not exp_src.is_dir():
            return []
        # Return paths relative to experiment/ (not codebase_dir) so that
        # when Coder.create() determines its root as experiment/, the file
        # paths don't get nested (e.g., experiment/experiment/config.py).
        return sorted(str(p.relative_to(exp_src)) for p in exp_src.rglob("*") if p.is_file() and self._should_include_file(p))

    def write_improvements(self, improvements: str) -> list[str]:
        """Write LLM-provided code changes, validating Python syntax.

        The improvements string contains file paths and new content for
        every file that should be modified.  Unchanged files are not included.

        Args:
            improvements: LLM output specifying files and new content.

        Returns:
            List of file paths that were successfully written.
        """
        if not improvements:
            _run_logger.warning("LLM returned no improvements")
            return []
        import evorun.utils.response as _utils
        parsed: list[dict[str, str]] = _utils.extract_jsons(improvements)
        if not parsed:
            _run_logger.warning("No JSON found in LLM response")
            return []
        written: list[str] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            content = item.get("code", "")
            path = item.get("file", item.get("path", ""))
            if not path or not content:
                continue
            try:
                full_path = self.codebase_dir / path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                # Only validate syntax for Python files; TOML is text-only.
                if not path.endswith(".py"):
                    full_path.write_text(content, encoding="utf-8")
                    written.append(path)
                    continue
                compile(content, path, "exec")
                full_path.write_text(content, encoding="utf-8")
                written.append(path)
            except SyntaxError as e:
                _run_logger.warning(f"Syntax error in {path}: {e}")
            except OSError as e:
                _run_logger.error(f"Failed to write {path}: {e}")
        return written

    def get_codebase_prompt(self, max_size: int = 4000) -> str:
        """Build a string with all Python files in experiment/.

        Used for prompt building so the LLM can see the full context.

        Args:
            max_size: Maximum characters to include (truncated).

        Returns:
            File contents formatted as "# File: <path>\n<content>".
        """
        result: list[str] = []
        total = 0
        exp_src = self.codebase_dir / "experiment"
        if not exp_src.is_dir():
            return ""
        for rel_path in sorted(p.relative_to(self.codebase_dir)
                                for p in exp_src.rglob("*.py")):
            try:
                abs_path = exp_src / rel_path
                content = abs_path.read_text(encoding="utf-8")
            except OSError:
                continue
            result.append(f"# File: {rel_path}\n{content}")
            total += len(content) + 30  # approximate
            if total > max_size:
                _run_logger.debug(f"Truncated codebase prompt at {max_size} chars")
                break
        return "\n".join(result)

    def restore_all(self) -> None:
        """Restore all files from the backup directory."""
        if not self.backup_dir.exists():
            _run_logger.warning("No backup to restore from")
            return
        for rel_path in self.backup_dir.rglob("*"):
            if rel_path.is_file():
                target = self.codebase_dir / rel_path.relative_to(self.backup_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rel_path, target)
        _run_logger.info(f"Restored all files from {self.backup_dir}")

    def _should_include_file(self, path: Path) -> bool:
        """Return True if *path* should be copied for backup/snapshots.

        Skips ``__pycache__`` directories and their contents."""
        return "__pycache__" not in path.parts


# ────────────────────────────────────────────────────────────
# LLM dispatch (Claude CLI only)
# ────────────────────────────────────────────────────────────


def _run_claude_cli(prompt: str, cwd: str, model: str = "claude-sonnet-4-6",
                    max_turns: int = 500, allowed_tools: list[str] | None = None,
                    log_file: str | None = None, retries: int = 3,
                    retry_base_delay: float = 5.0) -> str:
    """Run the claude CLI and return its output.

    Args:
        prompt: User prompt text.
        cwd: Working directory for the claude CLI.
        model: Claude model name.
        max_turns: Maximum tool-use turns.
        allowed_tools: List of allowed tool names.
        log_file: Optional file path to append output to.
        retries: Number of retry attempts on failure.
        retry_base_delay: Base delay in seconds for exponential backoff.

    Returns:
        LLM response text.
    """
    import time

    def _do_call():
        cmd = [
            'claude',
            '-p', prompt,
            '--output-format', 'text',
            '--max-turns', str(max_turns),
            '--model', model,
        ]
        if allowed_tools:
            cmd.extend(['--allowedTools'] + allowed_tools)

        log_fh = open(log_file, "w", encoding="utf-8") if log_file else None
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)
                if log_fh:
                    log_fh.write(line)
        finally:
            if log_fh:
                log_fh.close()
        process.wait()
        result = "".join(output_lines)
        if not result:
            raise RuntimeError("claude CLI returned empty output")
        return result

    for attempt in range(1, retries + 1):
        try:
            return _do_call()
        except Exception as e:
            _run_logger.warning(
                f"Claude CLI failed (attempt {attempt}/{retries}): {e}"
            )
            if attempt < retries:
                time.sleep(retry_base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"Claude CLI failed after {retries} retries")


_UNSET = object()  # sentinel for "explicitly unset this env var"


def _build_claude_env(provider: str, model: str, api_key: str | None) -> dict[str, object]:
    """Build env var overrides for the claude CLI based on provider.

    Returns a dict where:
    - string values override the env var
    - _UNSET means explicitly delete the env var
    """
    env: dict[str, object] = {}
    if provider == "anthropic":
        env["ANTHROPIC_BASE_URL"] = _UNSET
        env["ANTHROPIC_API_KEY"] = _UNSET
        env["ANTHROPIC_MODEL"] = _UNSET
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = _UNSET
    elif provider == "deepseek":
        env["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"
        env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_MODEL"] = model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    elif provider == "openrouter":
        env["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_MODEL"] = model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    else:
        # URL mode
        env["ANTHROPIC_BASE_URL"] = provider
        env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_MODEL"] = model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    return env


def _merge_env(env_overrides: dict[str, object]) -> dict[str, str]:
    """Merge env var overrides with current os.environ, handling _UNSET."""
    env = os.environ.copy()
    for key, value in env_overrides.items():
        if value is _UNSET:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def _run_claude_cli_with_env(
    prompt: str,
    cwd: str,
    model: str,
    env_overrides: dict[str, object],
    max_turns: int = 500,
    allowed_tools: list[str] | None = None,
    log_file: str | None = None,
    retries: int = 3,
    retry_base_delay: float = 5.0,
) -> str:
    """Run claude CLI with env var overrides (for provider switching)."""
    import time

    merged_env = _merge_env(env_overrides)

    def _do_call():
        cmd = [
            'claude',
            '-p', prompt,
            '--output-format', 'text',
            '--max-turns', str(max_turns),
            '--model', model,
        ]
        if allowed_tools:
            cmd.extend(['--allowedTools'] + allowed_tools)

        log_fh = open(log_file, "w", encoding="utf-8") if log_file else None
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
        )
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                output_lines.append(line)
                if log_fh:
                    log_fh.write(line)
        finally:
            if log_fh:
                log_fh.close()
        process.wait()
        result = "".join(output_lines)
        if not result:
            raise RuntimeError("claude CLI returned empty output")
        return result

    for attempt in range(1, retries + 1):
        try:
            return _do_call()
        except Exception as e:
            _run_logger.warning(
                f"Claude CLI failed (attempt {attempt}/{retries}): {e}"
            )
            if attempt < retries:
                time.sleep(retry_base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"Claude CLI failed after {retries} retries")


def _load_config_file() -> dict:
    """Load planner/editor config from ~/.config/evorun/evorun.toml."""
    config_path = os.path.expanduser("~/.config/evorun/evorun.toml")
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        _run_logger.warning(f"Failed to load evorun config: {e}")
        return {}


# ────────────────────────────────────────────────────────────
# EvoRunAgent
# ────────────────────────────────────────────────────────────


class EvoRunAgent:
    """Main iterative optimization loop with Monte Carlo Tree Search.

    Ties together Evaluator, CodebaseManager, LLM integration, and TreeSearch
    to run the full optimization cycle.

    Attributes:
        codebase: CodebaseManager for scanning, backing up, and restoring files.
        evaluator: Evaluator for running the evaluation command.
        tree: TreeSearch for node selection, expansion, and state persistence.
        coder: Claude CLI instance for LLM-driven code improvement.
        history: List of HistoryEntry records (one per iteration).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize EvoRunAgent with parsed arguments.

         Args:
            args: Parsed CLI arguments from parse_args().
        """
        self.codebase = CodebaseManager(
            args.codebase_dir,
            task_file="TASK.md",
        )
        eval_cmd = getattr(args, 'eval_cmd', None) or "pixi run python eval.py"
        self.evaluator = Evaluator(
            eval_cmd, args.eval_timeout,
            codebase_dir=self.codebase.codebase_dir,
        )

        self.fake_run = getattr(args, 'fake_run', False)

        # LLM config from CLI args, config file, and environment variables.
        self.model_name = getattr(args, 'model', None) or "claude-sonnet-4-6"

        # Load planner/editor config from ~/.config/evorun/evorun.toml
        config = _load_config_file()
        planner_cfg = config.get("planner", {})
        editor_cfg = config.get("editor", {})

        self.planner_model = planner_cfg.get("model") or self.model_name
        self.editor_model = editor_cfg.get("model") or self.planner_model

        self.planner_provider = planner_cfg.get("provider") or "anthropic"
        self.editor_provider = editor_cfg.get("provider") or "anthropic"

        self.planner_api_key = planner_cfg.get("api_key", None)
        self.editor_api_key = editor_cfg.get("api_key", None)

        self.editor_model_name = self.editor_model

        # Tree search max children
        self.tree_max_children = getattr(args, 'max_children', 10)
        self.optim_mode = getattr(args, 'optim_mode', 'max')
        self.tree = TreeSearch(maximize=(self.optim_mode == "max"), max_children=self.tree_max_children)
        self.tree.explore_c = self._get_decay_exploration_c()

        # Track current tree node snapshot for LLM feedback
        self._current_restored_node_id: str | None = None
        self.tree_expansion_count = 0

        # Background thread pool for LLM summary generation
        self._summary_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._summary_future: concurrent.futures.Future | None = None
        self._summary_lock = threading.Lock()

        # Optimization parameters
        self.max_iters: int = args.max_iters
        self.time_limit: float | None = args.time_limit
        self.patience: int = args.patience
        self.eval_timeout: int = args.eval_timeout
        self.eval_timeout_chain_limit: int = 3

        self.best_score: float | None = None
        self._best_snapshot_dir: str | None = None
        self._stop = False
        self.history: list[HistoryEntry] = []
        self.consecutive_timeouts: int = 0

        # Per-parent stagnation tracking (overrides global counter).
        # Keyed by parent node ID — tracks consecutive non-improving children.
        self._parent_stagnation: dict[str, int] = {}
        # Legacy per-branch tracking (kept for state file backward compat).
        self._branch_stagnation: dict[int, int] = {}

        # State and resume
        self.state_path = self.codebase.codebase_dir / ".evorun_state.json"
        self._resumed = False
        self._history_start: int = 0
        self._initial_hashes: dict[str, str] = {}

        # Last iteration file changes (for feedback context).
        self._last_modified_files: list[str] = []
        self._last_added_files: list[str] = []
        self._last_deleted_files: list[str] = []
        self._consecutive_no_changes: int = 0

        self.use_decay = getattr(args, 'decay_exploration', False)
        self.use_fusion = getattr(args, 'use_fusion', True)
        self.fusion_min_iters = getattr(args, 'fusion_min_iters', 10)
        self.fusion_prob = getattr(args, 'fusion_prob', 0.5)

        # Solution deduplication
        self._seen_code_hashes: set[str] = set()

        # Start the wall-clock timer.
        self._start_time = time.time()

        # Signal handler: exit immediately on Ctrl+C / SIGTERM.
        def _signal_handler(signum, frame) -> None:
            _run_logger.info("[Signal] Received SIGINT/SIGTERM, exiting.")
            sys.exit(1)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    def _attempt_resume(self) -> bool:
        """Try to resume from a saved state file.

        Returns:
            True if state was found and restored, False otherwise.
        """
        if not self.state_path.exists():
            return False
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            _run_logger.warning(f"Failed to read state file: {e}")
            return False
        # Parse history entries.
        self._history_start = data.get("next_iteration", 0)
        all_history = []
        for e in data.get("history", []):
            h = HistoryEntry(
                iter=e["iter"], score=e.get("score"), timed_out=e.get("timed_out", False),
                exec_time=e.get("exec_time", 0.0),
                datetime=e.get("datetime", ""),
                files_modified=e.get("files_modified", []),
                files_added=e.get("files_added", []),
                files_deleted=e.get("files_deleted", []),
                llm_response=e.get("llm_response", ""),
                edit_summary=e.get("edit_summary", ""),
                diff_text=e.get("diff_text", ""),
            )
            h.claude_log = e.get("claude_log", "")
            h.claude_feedback = e.get("claude_feedback", "")
            all_history.append(h)
        # Only keep history from the current session (iterations < next_iteration)
        # to prevent stale entries from triggering early stagnation detection.
        self.history = [h for h in all_history if h.iter < self._history_start]
        if len(all_history) > len(self.history):
            _run_logger.info(f"[State] Discarded {len(all_history) - len(self.history)} stale history entries from previous sessions")

        self.best_score = data.get("best_score")
        self._best_snapshot_dir = data.get("best_snapshot_iteration")
        self.consecutive_timeouts = data.get("consecutive_timeouts", 0)

        # Restore per-parent stagnation tracking (primary).
        parent_stag_data = data.get("parent_stagnation", {})
        self._parent_stagnation = {k: v for k, v in parent_stag_data.items()}
        # Restore legacy per-branch stagnation tracking (backward compat).
        self._branch_stagnation = {}
        branch_stag_data = data.get("branch_stagnation", {})
        self._branch_stagnation = {
            int(k) if str(k).isdigit() else k: v for k, v in branch_stag_data.items()
        }
        # Seed history for LLM (compact)
        for entry in self.history:
            llm = entry.llm_response.strip()
            if len(llm) > 300:
                llm = llm[:MAX_LLM_RESPONSE_SUMMARY] + "..."
            summary = entry.edit_summary if entry.edit_summary else "[N/A]"

        # Restore tree search state.
        ts_data = data.get("tree_structure")
        if ts_data:
            self.tree._restore_from_state(ts_data)
            _run_logger.info(
                f"[State] Tree restored: {len(ts_data['nodes'])} nodes, "
                f"max_depth={ts_data.get('max_depth', 0)}"
            )
        else:
            _run_logger.info("[State] Tree not found in state — starting fresh.")

        self._resumed = True
        return True

    def _build_claude(self) -> None:
        """Initialize the Claude CLI editor.

        The claude CLI reads ANTHROPIC_API_KEY from the environment.
        """
        self.editor_model_name = self.editor_model
        _run_logger.info(f"Using Claude CLI with model: {self.editor_model_name}")

# ─── Solution deduplication ─────────────────────────────────────

    def _compute_code_hash(self, code: str) -> str:
        """Compute SHA-256 hash of code for deduplication.

        Args:
            code: The code string to hash.

        Returns:
            SHA-256 hex digest of the code.
        """
        return hashlib.sha256(code.encode()).hexdigest()

    def _is_duplicate_code(self, code: str) -> bool:
        """Check if code is a duplicate of any previous solution.

        Args:
            code: The code to check.

        Returns:
            True if code is a duplicate.
        """
        code_hash = self._compute_code_hash(code)
        return code_hash in self._seen_code_hashes

    def _check_duplicate_on_disk(self) -> tuple[bool, str]:
        """Check if the current on-disk code is a duplicate of any previous solution.

        Reads only experiment files (not snapshots or other directories).

        Returns:
            (is_duplicate, code_string) — code_string is empty if duplicate or no .py files.
        """
        code_parts: list[str] = []
        for rel_path in self.codebase.get_experiment_files():
            file_path = self.codebase.codebase_dir / "experiment" / rel_path
            try:
                code = file_path.read_text(encoding="utf-8")
                code_parts.append(f"# File: {rel_path}\n{code}")
            except Exception:
                pass
        code = "\n".join(code_parts)
        if not code:
            return False, ""
        return self._is_duplicate_code(code), code

    # ─── Early exit conditions ─────────────────────────────────────

    def _check_topk_stagnation(self, topk: int = 5) -> bool:
        """Check if top-k candidates show stagnation (no improvement).

        Args:
            topk: Number of top candidates to check.

        Returns:
            True if top-k candidates are stagnant.
        """
        # Disabled
        return False

    # ─── Progressive UCT decay ─────────────────────────────────────

    def _get_decay_exploration_c(self) -> float:
        """Get the current exploration constant with progressive decay.

        Uses a piecewise decay schedule that reduces exploration over time.
        Early in the search, exploration is high; late in the search, exploitation dominates.

        Returns:
            Current exploration constant value.
        """
        if not getattr(self, 'use_decay', False):
            return 0.15  # Default static value

        elapsed = time.time() - self._start_time
        total_time = self.time_limit if self.time_limit is not None else 1e9
        progress = min(elapsed / total_time, 1.0)

        # Piecewise decay: start high, decay to lower bound
        start_c = 0.3
        end_c = 0.05
        decay_rate = 2.0  # Exponential decay rate

        # Apply exponential decay
        current_c = end_c + (start_c - end_c) * (1 - progress) ** decay_rate

        return max(current_c, end_c)

    # ─── Debug agent ─────────────────────────────────────

    def _run_debug_agent(self, result: EvalResult, target_node) -> str:
        """Run a simplified debug agent to analyze errors and provide fix instructions.

        Extracts error information from the eval result and generates targeted
 fix instructions for Claude.

        Args:
            result: Eval result with error information.
            target_node: The node that was expanded.

        Returns:
            Debug instructions string for Claude prompt, or empty string if no debug needed.
        """
        if not result.timed_out and not result.had_error and result.exit_code == 0:
            return ""

        debug_instructions = ["## Debug Instructions\n"]

        if result.timed_out:
            debug_instructions.append("**Issue**: Evaluation timed out")
            debug_instructions.append("**Action**: Check for infinite loops, expensive operations, or blocking I/O")
            debug_instructions.append("**Suggestions**:")
            debug_instructions.append("- Add timeout guards to loops")
            debug_instructions.append("- Remove blocking I/O calls")
            debug_instructions.append("- Optimize expensive operations")
        elif result.had_error:
            debug_instructions.append("**Issue**: Evaluation failed with error")
            debug_instructions.append("**Action**: Fix the error based on the traceback below")
        elif result.exit_code != 0:
            debug_instructions.append(f"**Issue**: Evaluation exited with code {result.exit_code}")
            debug_instructions.append("**Action**: Check the error output and fix the issue")

        if result.output:
            debug_instructions.append("\n### Error Output\n")
            debug_instructions.append("```")
            for line in result.output[:20]:
                debug_instructions.append(str(line)[:200])
            debug_instructions.append("```")

        return "\n".join(debug_instructions)

    # ─── Code review agent ─────────────────────────────────────

    def _run_code_review(self, code: str) -> tuple[bool, str]:
        """Run a simplified code review check on the given code.

        Checks for common issues like data leakage, hardcoded values, and
        missing error handling. Returns (needs_fix, review_notes).

        Args:
            code: The code to review.

        Returns:
            Tuple of (needs_fix: bool, review_notes: str).
        """
        review_notes = []

        # Check for data leakage patterns
        leakage_patterns = [
            (r"test[_.]data", "Potential data leakage: test data reference"),
            (r"validation[_.]data", "Potential data leakage: validation data reference"),
            (r"shuffle\(.*seed", "Potential data leakage: shuffled with fixed seed"),
        ]

        for pattern, message in leakage_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                review_notes.append(message)

        # Check for hardcoded values
        if re.search(r"score\s*=\s*\d+\.\d+", code):
            review_notes.append("Warning: Hardcoded score value detected")
        if re.search(r"target\s*=\s*\d+\.\d+", code):
            review_notes.append("Warning: Hardcoded target value detected")

        # Check for missing error handling
        if "try:" not in code and "except" not in code:
            review_notes.append("Warning: No error handling detected")

        needs_fix = len(review_notes) > 0
        return needs_fix, "\n".join(review_notes)

    def _run_planner(self, feedback: str) -> str:
        """Run the planner stage: analyze feedback and produce a free-form plan.

        The planner reads the full feedback prompt and outputs a natural-language
        plan describing WHAT files to modify and WHAT changes to make. It does NOT
        write code.

        Args:
            feedback: The formatted feedback string from _format_feedback().

        Returns:
            Free-form text plan or empty string if planning fails.
        """
        planner_env = _build_claude_env(self.planner_provider, self.planner_model, self.planner_api_key)
        planner_prompt = f"""\
You are a code planning architect. Your job is to analyze evaluation results
and code review feedback, then produce a plan for what changes to make.

IMPORTANT RULES:
1. Do NOT write any code. Do NOT use Read, Edit, or Write tools.
2. Do NOT produce diffs or file content.
3. Describe changes in plain language — the editor will implement them.

Here is the feedback to analyze:
{feedback}

Produce a concise plan describing what to change and why.
"""

        try:
            result = _run_claude_cli_with_env(
                prompt=planner_prompt,
                cwd=str(self.codebase.codebase_dir),
                model=self.planner_model,
                env_overrides=planner_env,
                max_turns=30,
                allowed_tools=[],
                retries=getattr(self, 'llm_retries', 3),
                retry_base_delay=getattr(self, 'llm_retry_base_delay', 3.0),
            )
            return result.strip()
        except Exception as e:
            _run_logger.warning(f"Planner failed (iter {self._iteration}): {e}")
            return ""

    def _run_editor(self, plan: str, feedback: str) -> str:
        """Run the editor stage: execute the plan by writing actual code.

        The editor receives the plan from the planner, reads the relevant files
        as needed, and makes code changes via Read/Edit/Write.

        Args:
            plan: The free-form plan text from _run_planner().
            feedback: The full feedback for context (scores, eval output, etc.).

        Returns:
            The raw claude CLI output (tool use logs).
        """
        editor_env = _build_claude_env(self.editor_provider, self.editor_model, self.editor_api_key)
        editor_prompt = f"""\
You are a senior Python developer implementing a code improvement plan.

## Your Plan
{plan}

## Context (from evaluation)
{feedback}

## Instructions
1. Read the files mentioned in the plan to understand the current code.
2. Implement each change described in the plan faithfully.
3. Make sure your changes produce valid, runnable Python code.
4. Preserve working code that is not mentioned in the plan.
5. Output complete files, not partial diffs.
6. Do NOT make changes beyond what the plan specifies.
7. Do not suggest EDA.
8. Focus on improving the evaluation score.
"""

        try:
            result = _run_claude_cli_with_env(
                prompt=editor_prompt,
                cwd=str(self.codebase.codebase_dir),
                model=self.editor_model,
                env_overrides=editor_env,
                max_turns=500,
                allowed_tools=['Read', 'Edit', 'Write'],
                log_file=str(self.codebase.codebase_dir / ".evorun_claude.log"),
                retries=getattr(self, 'llm_retries', 3),
                retry_base_delay=getattr(self, 'llm_retry_base_delay', 3.0),
            )
            return result
        except Exception as e:
            _run_logger.warning(f"Editor failed (iter {self._iteration}): {e}")
            return ""

    # ─── Main entry point ────────────────────────────────────────

    def run(self) -> None:
        """Execute the full optimization loop with Monte Carlo Tree Search.

        Main loop that iteratively explores the search tree:
            1.  Selects a node to expand — UCT across entire tree, not just root
            2.  Restores that node's snapshot to disk
            3.  Runs baseline eval (only on first visit or if node has no score)
            4.  Sends feedback to Claude CLI for improvement
            5.  Runs eval to get the child node's score
            6.  Creates child node, always saves snapshot, backpropagates reward
            7.  Tracks best score and per-branch stagnation
            8.  Stops when ALL branches are stagnant (not just any single counter)

        Returns:
            None.  Exit status is logged and best snapshot is restored.
    """
        import evorun.utils.response as utils
        # Seed and initialize.
        self._attempt_resume()
        if not self.fake_run:
            self._build_claude()

        # Setup codebase / snapshots / state.
        if not self._resumed:
            self.codebase.setup(resuming=False)
            self._initial_hashes = self._compute_file_hashes(
                self.codebase.get_experiment_files(),
            )
        else:
            self.codebase.setup(resuming=True)
            self._initial_hashes = self._compute_file_hashes(
                self.codebase.get_experiment_files(),
            )

        if not self.fake_run:
            _run_logger.info("=" * 72)
            _run_logger.info("[EvoRunAgent] Starting optimization loop.")
            _run_logger.info("=" * 72)

        self._iteration = self._history_start

        while self._iteration < self.max_iters and not self._stop:
            wall_elapsed = time.time() - self._start_time
            if self.time_limit is not None and wall_elapsed > self.time_limit:
                _run_logger.info(f"[Stop] Elapsed time ({wall_elapsed:.0f}s) exceeded "
                            f"time limit ({self.time_limit}s).")
                break

            if self._iteration > 0:
                _run_logger.info(f"[Iter {self._iteration}] "
                            f"--- Next iteration ---")

            self.tree_expansion_count += 1

            try:
                self._run_iteration()
            except Exception as e:
                _run_logger.error(
                    f"[Iter {self._iteration}] Unexpected error: {e}",
                    exc_info=True,
                )
                self.history.append(HistoryEntry(
                    iter=self._iteration, score=None,
                    timed_out=False, exec_time=0.0,
                    datetime=datetime.now().isoformat(),
                    files_modified=[], files_added=[], files_deleted=[],
                    llm_response="", edit_summary=f"Error: {e}", diff_text="",
                ))
                self._iteration += 1
                self.save_state()

        # Restore best snapshot on exit.
        if self.fake_run:
            _run_logger.info("[Mode] Fake run — no eval or LLM calls, just random scores")

        if self._best_snapshot_dir:
            _run_logger.info(f"[Exit] Restoring best checkpoint...")
            self._restore_snapshot(self._best_snapshot_dir)
        else:
            _run_logger.info("[Exit] No valid scores achieved. Codebase unchanged.")

    def _run_iteration(self):
        """Execute the body of one optimization iteration.

        Called from the ``run()`` while-loop.  Contains Steps 0-6 of the
        optimization loop: node selection, snapshot restore, baseline eval,
        debug / review / code generation, child creation, stagnation
        tracking, and state persistence.
        """
        # ---- Step 0: Select node to expand (UCT across entire tree) ----
        target_branch_id = len(self.tree.root.children) + 1
        target_node = self.tree.select_node(branch_id=target_branch_id)
        # avg_reward = current UCT exploration value (visits==1 → inf).
        avg_reward = (
            target_node.total_reward / max(target_node.visits, 1)
            if target_node.visits else float("inf")
        )
        _run_logger.info(
            f"[Iter {self._iteration + 1}/{self.max_iters}] "
            f"Tree search: selected node {target_node.id[:8]} "
            f"(Q={avg_reward:.4f}, visits={target_node.visits}, "
            f"children={target_node.num_children}, depth={self._node_depth(target_node)})"
        )

        # Update decay exploration constant
        if self.use_decay:
            self.tree.explore_c = self._get_decay_exploration_c()

        # Check if ALL nodes are at their child limit (dead tree).
        all_blocked = all(
            n.num_children >= self.tree.max_children
            for n in self.tree.journal.nodes
            if n is not self.tree.root
        )
        if all_blocked and self.tree.root.num_children >= self.tree.max_children:
            _run_logger.info(f"[Stop] No expandable nodes left — dead tree after "
                        f"{len(self.tree.journal)} nodes.")
            self._stop = True
            return

        # ---- Step 1: Restore target node's snapshot to disk ----
        need_baseline = False
        # Check if the target node has never been evaluated — if so, run
        # baseline eval.  Once a node has a metric (even with value=None
        # from a timeout/error), it is never re-evaluated.
        if target_node.metric is None:
            need_baseline = True
        else:
            _score = target_node.metric.value
            _best = self.best_score if self.best_score is not None else "N/A"
            _run_logger.info(f"[reuse] node={target_node.id[:8]} score={_score} best={_best}")
            _run_logger.info(
                f"[Iter {self._iteration}] Reusing existing eval score "
                f"for node {target_node.id[:8]} (skip baseline)"
            )
        target_node_dir: str | None = None
        if target_node is not self.tree.root:
            target_node_dir = self._find_node_snapshot(target_node)
        if target_node_dir is not None:
            self._restore_snapshot(target_node_dir)
            self._current_restored_node_id = target_node.id
        else:
            self._current_restored_node_id = None

        if target_node is self.tree.root:
            self._current_restored_node_id = None

        # Ensure target_node has a snapshot for later diff computation.
        # Child expansions will diff against this snapshot, so it must
        # exist before we call Claude.
        if target_node is not self.tree.root:
            target_snap = self._find_node_snapshot(target_node)
            if not target_snap:
                self._save_node_snapshot(target_node, self._iteration)
                _run_logger.info(
                    f"[Snapshot] Created snapshot for node "
                    f"{target_node.id[:8]} (needed for diff computation)"
                )

        # ---- Step 2: Run baseline eval ----
        # Only if the current node doesn't have a score yet.
        result = EvalResult()

        if need_baseline:
            if self.fake_run:
                fake_score = self._generate_fake_score()
                result = EvalResult(
                    score=fake_score, description=f"Fake score (iteration {self._iteration})",
                    timed_out=False, exec_time=0.05, output=[],
                )
                _run_logger.info(f"[Iter {self._iteration}/{self.max_iters}] Fake eval: "
                            f"score={result.score:.4f}")
                self.consecutive_timeouts = 0
            else:
                result = self.evaluator.run()
                _run_logger.info(f"[Iter {self._iteration}/{self.max_iters}] Baseline eval: "
                            f"score={result.score} (timed_out={result.timed_out}, "
                            f"{result.exec_time:.1f}s)")

                if result.timed_out:
                    self.consecutive_timeouts += 1
                    if self.consecutive_timeouts >= self.eval_timeout_chain_limit:
                        _run_logger.warning(
                            f"Eval timed out {self.consecutive_timeouts} "
                            f"times in a row — treating timeouts as eval errors "
                            "and continuing (Claude will still be called)."
                        )
                else:
                    self.consecutive_timeouts = 0

            # Store eval output on the node so Claude can see it.
            target_node._term_out = result.output

            # Update global best score and best snapshot.
            if result.score is not None:
                improved = False
                if self.best_score is None:
                    improved = True
                elif self.tree.maximize and result.score > self.best_score:
                    improved = True
                elif not self.tree.maximize and result.score < self.best_score:
                    improved = True
                if improved:
                    self.best_score = result.score
                    self._best_snapshot_dir = self._save_node_snapshot(target_node, self._iteration)
                    _run_logger.info(f"[Iter {self._iteration}] New best score: {result.score}")

            # If this is the root node (first baseline only), store the score
            # so subsequent visits skip baseline and proceed to child creation.
            if target_node is self.tree.root and result.score is not None:
                target_node.metric = MetricValue(
                    value=result.score, maximize=self.tree.maximize
                )
                # Save state after recording root baseline (no history entry —
                # the first child expansion at the same iter=0 would collide).
                self.save_state()
            # Save root snapshot after baseline eval so child diffs
            # have a baseline to compare against.
            root_snap = self._find_node_snapshot(target_node)
            if not root_snap:
                self._save_node_snapshot(target_node, self._iteration)
                _run_logger.info(
                    "[Snapshot] Created snapshot for root node "
                    "(needed for diff computation)"
                )
        else:
            # Node was re-expanded with existing score — reuse existing score from node.
            # Preserve stagnation from previous children — re-expanding a
            # parent should not reset the branch's accumulated stagnation
            # since children share the same branch_id.
            score_val = target_node.metric.value  # type: ignore
            _best = self.best_score if self.best_score is not None else "N/A"
            _run_logger.info(f"[reuse] node={target_node.id[:8]} score={score_val} best={_best}")
            result = EvalResult(
                score=score_val,
                description=f"Reused from node {target_node.id[:8]}",
                timed_out=False,
                exec_time=getattr(target_node, 'exec_time', None) or 0.0,
                output=[],
        )

        # ---- Step 3: Run debug agent (if eval failed) ----
        debug_instructions = ""
        if not self.fake_run and (result.timed_out or result.had_error or (result.exit_code is not None and result.exit_code != 0)):
            debug_instructions = self._run_debug_agent(result, target_node)

        # ---- Step 4: Run code review (if enabled) ----
        code_review_notes = ""
        if not self.fake_run and target_node.code:
            needs_fix, notes = self._run_code_review(target_node.code)
            if needs_fix and notes:
                code_review_notes = f"## Code Review\n{notes}\n"

        # Compute parent_id early — used by stagnation tracking.
        # Stagnation is keyed by the node being expanded (target_node.id),
        # since we are tracking how many non-improving children it produces.
        parent_id = target_node.id

        # ---- Step 5: Code generation (fusion or normal Claude) ----
        tree_info = self.tree.get_node_info()
        # Always pass eval output so LLM can debug from error traces.
        # Combine stored _term_out (from prior expansions) with the
        # latest result.output (baseline eval just ran).
        prev_eval_parts = []
        stored = getattr(target_node, "_term_out", None)
        if stored:
            if isinstance(stored, list):
                prev_eval_parts.append("\n".join(str(x) for x in stored))
            else:
                prev_eval_parts.append(str(stored))
        if result.output:
            prev_eval_parts.append("\n".join(str(x) for x in result.output))
        prev_eval = "\n".join(prev_eval_parts)

        log_content: str = ""
        modified_files: list[str] = []
        added_files: list[str] = []
        deleted_files: list[str] = []
        any_changes = False
        prev_hashes: dict[str, str] = {}
        used_fusion = False

        if not self.fake_run and self.use_fusion:
            # Only use fusion after fusion_min_iters iterations
            if self._iteration >= self.fusion_min_iters:
                do_fusion = random.random() < self.fusion_prob
            else:
                do_fusion = False
            if do_fusion:
                # Try cross-branch fusion first
                _run_logger.info(f"[Iter {self._iteration}] Trying cross-branch fusion...")
                _, log_content, modified_files, added_files, deleted_files, used_fusion = \
                    self._run_fusion(target_node)

        # Build feedback (always, for logging and next iteration)
        parent_score = target_node.parent.metric.value if target_node.parent and target_node.parent.metric else None
        feedback = self._format_feedback(result, self._iteration, tree_info, prev_eval, code_review_notes, debug_instructions, parent_score)

        if not (used_fusion and log_content) and not self.fake_run:
            # Two-stage: planner → editor
            if feedback:
                _run_logger.info(f"[Iter {self._iteration}] Sending feedback to planner...")
                prev_hashes = self._compute_file_hashes(
                    self.codebase.get_experiment_files(),
                )

                plan = self._run_planner(feedback)
                if plan and plan.strip():
                    _run_logger.info(f"[Iter {self._iteration}] Running editor with plan...")
                    log_content = self._run_editor(plan, feedback)
                    if log_content:
                        _run_logger.info(
                            f"[Claude] Editor output: {len(log_content)} chars"
                        )
                else:
                    _run_logger.warning(
                        f"[Iter {self._iteration}] Planner produced empty plan — skipping code gen"
                    )
                    log_content = ""
            # If feedback is empty, log_content stays empty (no LLM call)

            # Detect actual file changes (hashes only).
            current_hashes = self._compute_file_hashes(
                self.codebase.get_experiment_files(),
            )
            # Files that exist now but not before → added.
            for fpath, chash in current_hashes.items():
                if fpath not in prev_hashes:
                    added_files.append(fpath)
            # Files that existed before but not now → deleted.
            for fpath in prev_hashes:
                if fpath not in current_hashes:
                    deleted_files.append(fpath)
            # Files that exist in both but changed → modified.
            for fpath, chash in current_hashes.items():
                if fpath in prev_hashes and prev_hashes[fpath] != chash:
                    modified_files.append(fpath)
            any_changes = modified_files or added_files or deleted_files
            if any_changes:
                _run_logger.info("[Claude] Improved code:")
                for f in modified_files:
                    _run_logger.info(f"  ~ {f}")
                for f in added_files:
                    _run_logger.info(f"  + {f}")
                for f in deleted_files:
                    _run_logger.info(f"  - {f}")
                self._consecutive_no_changes = 0
            else:
                _run_logger.info("[Claude] No changes detected — skipped")
                self._consecutive_no_changes += 1
                if self._consecutive_no_changes >= MAX_CONSECUTIVE_NO_CHANGES:
                    _run_logger.warning(
                        f"[Claude] No changes for {self._consecutive_no_changes} "
                        f"consecutive iterations. Check .evorun_claude.log for debug info."
                    )
            # Store for next iteration's feedback.
            self._last_modified_files = modified_files
            self._last_added_files = added_files
            self._last_deleted_files = deleted_files

        # Track fusion path file changes and stagnation
        if used_fusion:
            if modified_files or added_files or deleted_files:
                _run_logger.info("[Fusion] Code modified:")
                for f in modified_files: _run_logger.info(f"  ~ {f}")
                for f in added_files: _run_logger.info(f"  + {f}")
                for f in deleted_files: _run_logger.info(f"  - {f}")
                self._consecutive_no_changes = 0
            else:
                _run_logger.info("[Fusion] No changes detected")
                self._consecutive_no_changes += 1
                if self._consecutive_no_changes >= MAX_CONSECUTIVE_NO_CHANGES:
                    _run_logger.warning(
                        f"[Fusion] No changes for {self._consecutive_no_changes} "
                        f"consecutive iterations."
                    )
            self._last_modified_files = modified_files
            self._last_added_files = added_files
            self._last_deleted_files = deleted_files
            any_changes = modified_files or added_files or deleted_files

        # Response verification logging
        if log_content:
            _run_logger.info(f"[Claude] Response received, length={len(log_content)}")
            _run_logger.debug(
                f"[Claude] Response preview: {log_content[:500]}..."
            )
        elif feedback and not self.fake_run:
            _run_logger.warning(f"[Claude] No response received from LLM")

        # Compute diff text synchronously (lightweight).
        diff_text = ""
        if any_changes and not self.fake_run:
            parent_snap = self._find_node_snapshot(target_node)
            if not parent_snap:
                _run_logger.warning(
                    f"[Iter {self._iteration}] Cannot compute diff: "
                    f"snapshot not found for node {target_node.id[:8]}. "
                    f"This should not happen - please report this."
                )
            if parent_snap:
                diff_text = self._compute_diff_from_snapshot(
                    parent_snap, modified_files, added_files, deleted_files,
                )

        # Submit edit summary to background thread — runs in parallel
        # with the child eval below.
        edit_summary = ""
        if diff_text:
            self._submit_edit_summary(diff_text, self._iteration)

        # Save the full LLM response to seed history (compact).
        llm_response = (log_content or "").strip()
        llm_response_short = llm_response

        # ---- Step 6: Run eval on improved code (child score) ----
        child_score: float | None = None
        child_node: Any | None = None
        child_result: EvalResult | None = None

        # Evaluate child when: (a) fake-run, or (b) Claude made changes.
        # Skip eval when there are no code changes — reuse parent's score.
        parent_score = (
            target_node.metric.value
            if target_node.metric and target_node.metric.value is not None
            else 0.0
        )
        no_changes = (
            not modified_files and not added_files and not deleted_files
        )
        if self.fake_run:
            # Generate fake child score based on parent's score.
            # Uniform random variation — unbiased for tree-search
            # prototyping (avoid depth bias, Issue 13).
            delta = random.uniform(-FAKE_SCORE_DELTA_RANGE, FAKE_SCORE_DELTA_RANGE)
            child_score = parent_score + delta
            _run_logger.info(f"[Iter {self._iteration}] Fake child score: "
                         f"{child_score:.4f}")
        elif no_changes:
            _run_logger.info(f"[Iter {self._iteration}] No code changes — "
                         f"reusing parent score {parent_score:.4f}")
            child_score = parent_score
        else:
            _run_logger.info(f"[Iter {self._iteration}] Evaluating "
                         f"child (~{len(modified_files)} modified, "
                         f"+{len(added_files)} added, "
                         f"-{len(deleted_files)} deleted)...")
            child_result = self.evaluator.run()
            child_score = child_result.score
            _run_logger.info(f"[Iter {self._iteration}] Child eval: "
                        f"score={child_score} ({child_result.exec_time:.1f}s)")
            # Try to get edit summary without blocking (runs in background).
            edit_summary = ""
            if self._summary_future is not None:
                try:
                    edit_summary = self._summary_future.result(timeout=5.0) or ""
                except concurrent.futures.TimeoutError:
                    pass
                except Exception:
                    pass
        # Record eval errors and timeouts (they count as scored as None, so
        # stagnation logic handles them below).
        if child_result and (child_result.had_error or child_result.timed_out):
            self.consecutive_timeouts += 1
        else:
            self.consecutive_timeouts = 0

        # ---- Save snapshot and check for duplicates BEFORE creating child node ----
        # This prevents duplicate nodes from being added to the tree.
        child_code = ""
        is_dup = False
        pre_snap_dir = None
        if not self.fake_run:
            # Save snapshot with a temporary name (before we know the child's ID).
            pre_snap_name = f"iter_snapshot_pre_{self._iteration}"
            pre_snap_dir = self._save_node_snapshot(
                target_node, self._iteration, dir_name=pre_snap_name,
            )
            _run_logger.info(
                f"[Snapshot] Saved pre-child snapshot "
                f"({pre_snap_name})"
            )

            # Check for duplicate code on disk.
            child_code = ""
            is_dup, child_code = self._check_duplicate_on_disk()
            if is_dup:
                child_code = ""  # clear for hash tracking below

        # If duplicate, skip child creation entirely (don't add to tree).
        if is_dup:
            _run_logger.info(
                f"[Iter {self._iteration}] Duplicate code detected — "
                f"skipping node creation"
            )
            # Clean up temp snapshot.
            if pre_snap_dir:
                try:
                    shutil.rmtree(pre_snap_dir)
                except OSError:
                    pass
            # Update stagnation — duplicate is a non-improving expansion.
            self._parent_stagnation[parent_id] = (
                self._parent_stagnation.get(parent_id, 0) + 1
            )
            # Check if ALL parents are stagnant.
            stagnant_count = sum(
                1 for v in self._parent_stagnation.values()
                if v >= self.patience
            )
            total_parents = len(self._parent_stagnation)
            if total_parents > 0 and stagnant_count >= total_parents:
                _run_logger.info(
                    f"[Stop] All {stagnant_count} parent(s) stagnant "
                    f">(={self.patience} consecutive non-improvement)."
                )
                self._stop = True
                return
            # Record as a skipped iteration
            self.history.append(HistoryEntry(
                iter=self._iteration, score=None,
                timed_out=False, exec_time=0.0,
                datetime=datetime.now().isoformat(),
                files_modified=[], files_added=[], files_deleted=[],
                llm_response="", edit_summary="", diff_text="",
            ))
            self._iteration += 1
            self.save_state()
            return

        # Not a duplicate — create the child node and finalize snapshot.
        child_output = (
            getattr(child_result, "output", [])
            if not self.fake_run and child_result else []
        )
        child_node = self.tree.make_child(
            parent=target_node,
            score=child_score,
            step=self._iteration,
            eval_output="\n".join(child_output),
        )
        child_node._term_out = child_output
        if used_fusion:
            child_node.stage = "fusion"

        # Populate code_summary on the child.
        # Use the edit summary if available, otherwise the file change list.
        if edit_summary:
            child_node.code_summary = edit_summary
        else:
            parts = []
            if modified_files:
                parts.append(f"modified: {', '.join(modified_files)}")
            if added_files:
                parts.append(f"added: {', '.join(added_files)}")
            if deleted_files:
                parts.append(f"deleted: {', '.join(deleted_files)}")
            child_node.code_summary = " ".join(parts) if parts else None

        _run_logger.info(f"[Iter {self._iteration}] Child created: "
                    f"node {child_node.id[:8]} with score={child_score}")

        # Log summary line
        if log_content:
            _run_logger.info(f"[Iter {self._iteration}] Summary: {edit_summary}")

        # Rename pre-snapshot to child's actual snapshot name.
        if pre_snap_dir:
            old_name = Path(pre_snap_dir)
            new_name = (
                self.codebase.codebase_dir / ".evorun_snapshots"
                / f"iter_snapshot_{child_node.id[:8]}"
            )
            if old_name.exists() and not new_name.exists():
                old_name.rename(new_name)
            elif old_name.exists():
                # New name already exists (shouldn't happen), remove temp.
                shutil.rmtree(old_name)
            _run_logger.info(
                f"[Snapshot] Finalized child snapshot "
                f"iter_snapshot_{child_node.id[:8]}"
            )

        # Track this code as seen.
        if child_code:
            self._seen_code_hashes.add(self._compute_code_hash(child_code))
        else:
            # Fallback: read from the finalized snapshot.
            final_code = self._read_snapshot_code(child_node)
            if final_code:
                self._seen_code_hashes.add(self._compute_code_hash(final_code))

        # Record HistoryEntry for this child expansion.
        self.history.append(HistoryEntry(
            iter=self._iteration, score=child_score,
            timed_out=child_result.timed_out if child_result else False,
            exec_time=child_result.exec_time if child_result else 0.0,
            datetime=datetime.now().isoformat(),
            files_modified=modified_files,
            files_added=added_files,
            files_deleted=deleted_files,
            llm_response=(log_content or "").strip(),
            edit_summary=edit_summary,
            diff_text=diff_text,
        ))
        self.history[-1].claude_log = log_content or ""
        self.history[-1].claude_feedback = feedback or ""

        # Update global best if child score improves (respects optim-mode).
        if child_score is not None:
            if self.best_score is None:
                improved = True
            else:
                improved = (child_score > self.best_score) if self.tree.maximize else (child_score < self.best_score)
            if improved:
                self.best_score = child_score
                self._best_snapshot_dir = f"iter_snapshot_{child_node.id[:8]}"
                _run_logger.info(f"[Iter {self._iteration}] New global best: "
                                 f"{child_score:.6f}")

        # Update per-parent stagnation for child score.
        # Tracks consecutive non-improving children per parent node.
        stagnation_parent_id = target_node.id
        parent_score = (
            target_node.metric.value if target_node.metric else None
        )

        if child_score is not None and parent_score is not None:
            improved = (child_score > parent_score) if self.tree.maximize else (child_score < parent_score)
            if improved:
                self._parent_stagnation[stagnation_parent_id] = 0
            else:
                self._parent_stagnation[stagnation_parent_id] = (
                    self._parent_stagnation.get(stagnation_parent_id, 0) + 1
                )
        elif child_score is None:
            # Claude produced no change or eval returned no score — stagnation.
            self._parent_stagnation[stagnation_parent_id] = (
                self._parent_stagnation.get(stagnation_parent_id, 0) + 1
            )

        # Check if ALL parents have been stagnant — stop entire run.
        # Only parents that have been expanded (produced children) are tracked.
        stagnant_count = sum(
            1 for v in self._parent_stagnation.values()
            if v >= self.patience
        )
        total_parents = len(self._parent_stagnation)
        if total_parents > 0 and stagnant_count >= total_parents:
            _run_logger.info(
                f"[Stop] All {stagnant_count} parent(s) stagnant "
                f">(={self.patience} consecutive non-improvement)."
            )
            self._stop = True
            return

        # Check top-k stagnation
        if self._check_topk_stagnation(topk=5):
            _run_logger.info(
                f"[Stop] Top-5 candidates stagnant — no improvement in recent iterations."
            )
            self._stop = True
            return

        # ---- Step 5: Save state ----
        self.save_state()
        self._iteration += 1

    def _node_depth(self, node: Any) -> int:
        """Return depth of node from root (root = 0)."""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    @staticmethod
    def _parse_eval_description(description: str) -> list[tuple[str, float]]:
        """Parse key=value and key: value pairs from an eval description string.

        Args:
            description: The "description" field from the eval output.

        Returns:
            List of (name, numeric_value) tuples.
        """
        metrics: list[tuple[str, float]] = []
        parts = description.replace(",", " ").split()
        for part in parts:
            parsed = False
            for sep in [":", "="]:
                if sep in part:
                    name, value = part.split(sep, 1)
                    try:
                        metrics.append((name.strip(), float(value)))
                        parsed = True
                        break
                    except ValueError:
                        continue
        return metrics

    @staticmethod
    def _extract_task_hints(task_content: str | None) -> list[str]:
        """Extract bullet hints from the Hints section of a task file.

        Args:
            task_content: Contents of TASK.md or None if not found.

        Returns:
            List of hint strings from the Hints section.
        """
        if not task_content:
            return []

        lines = task_content.splitlines()
        hints = []
        in_hints = False

        for line in lines:
            if re.match(r'^##\s+Hints\b', line):
                in_hints = True
                continue
            if in_hints:
                if line.strip().startswith("#"):
                    break
                if line.strip().startswith("-"):
                    hints.append(line.strip()[1:].strip())

        return hints

    def _generate_fake_score(self) -> float:
        """Generate a fake score for the current node (fake-run only).

        Returns a uniform random score in [0, 1] so that tree-search
        hyper-parameter tuning is not biased by artificial depth-correlation.

        Returns:
            Fake score in [0, 1] range.
        """
        return random.uniform(0.0, 1.0)

    def save_state(self) -> None:
        """Save the current optimization state to disk.

        Writes the following to .evorun_state.json:
            - next_iteration
            - best_score, best_snapshot_iteration
            - consecutive_timeouts
            - history (last 20 entries, truncated llm_response to 300 chars)
            - tree state (root_id, best_node_id, branch_stagnation,
                branch_best, full tree_structure)
       """
        history_entries: list[dict[str, Any]] = []
        for e in self.history:
            history_entries.append({
                "iter": e.iter, "score": e.score, "timed_out": e.timed_out,
                "exec_time": e.exec_time, "datetime": e.datetime,
                "files_modified": e.files_modified,
                "files_added": e.files_added,
                "files_deleted": e.files_deleted,
                "llm_response": e.llm_response[:MAX_LLM_RESPONSE_SUMMARY],
                "edit_summary": e.edit_summary,
                "diff_text": e.diff_text,
                "claude_log": getattr(e, "claude_log", ""),
                "claude_feedback": getattr(e, "claude_feedback", ""),
            })
        # Derive next_iteration from tree structure to stay in sync with history
        max_step = max((n.step for n in self.tree.journal.nodes), default=0)
        data = {
            "next_iteration": max_step + 1,
            "best_score": self.best_score,
            "best_snapshot_iteration": self._best_snapshot_dir,
            "consecutive_timeouts": self.consecutive_timeouts,
            "history": history_entries,
        }
        # Include parent stagnation (primary) and branch stagnation (legacy).
        parent_stagnation = {k: v for k, v in self._parent_stagnation.items()}
        data["parent_stagnation"] = parent_stagnation
        branch_stagnation = {str(k): v for k, v in self._branch_stagnation.items()}
        data["branch_stagnation"] = branch_stagnation
        # Full tree state.
        data["tree_structure"] = self.tree.get_tree_structure()

        # Atomic write: dump to a temp file in the same directory, then
        # rename (os.replace is atomic on POSIX).  This prevents a crashed
        # run from leaving a half-written state file.
        dir_name = str(self.state_path.parent)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp_path, self.state_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        _run_logger.info(f"[State] Saved to {self.state_path} "
                    f"(next_iter={max_step+1}, best_score={self.best_score})")

    # ─── Snapshot helpers ────────────────────────────────────────

    def _save_node_snapshot(self, node: Any, step: int, *, dir_name: str | None = None) -> str:
        """Save the current codebase as a snapshot for this tree node.

        Backs up experiment/ (all files) + pixi.toml.
        Node snapshots are named iter_snapshot_<node.id[:8]>.
        Also records which files were deleted compared to the parent snapshot.

        Args:
            node: The SearchNode this snapshot belongs to.
            step: Iteration step number.
            dir_name: Optional explicit snapshot directory name (bypasses node.id naming).

        Returns:
            Path string to the snapshot directory.
        """
        if dir_name:
            target_name = dir_name
        else:
            target_name = f"iter_snapshot_{node.id[:8]}"
        snapshot_dir = (
            self.codebase.codebase_dir / ".evorun_snapshots" / target_name
        )

        # Collect files that exist in the parent snapshot (if any).
        # Must happen BEFORE deleting the current snapshot directory,
        # and must look up the PARENT node's snapshot, not our own.
        parent_deleted: list[str] = []
        parent_snap = (
            self._find_node_snapshot(node.parent)
            if node.parent else None
        )

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if parent_snap:
            parent_snap_path = Path(parent_snap)
            parent_files = set()
            for p in parent_snap_path.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(parent_snap_path)
                    parent_files.add(str(rel))
            # Current files.
            current_files: set[str] = set()
            exp_src = self.codebase.codebase_dir / "experiment"
            if exp_src.is_dir():
                for p in exp_src.rglob("*"):
                    if p.is_file() and self.codebase._should_include_file(p):
                        rel = str(p.relative_to(self.codebase.codebase_dir))
                        current_files.add(rel)
            # Files in parent but not in current → deleted.
            for rel in parent_files:
                if rel not in current_files:
                    parent_deleted.append(rel)

        # Copy experiment/ subfolder (all files)
        exp_src = self.codebase.codebase_dir / "experiment"
        if exp_src.is_dir():
            for p in exp_src.rglob("*"):
                if p.is_file() and self.codebase._should_include_file(p):
                    target = snapshot_dir / p.relative_to(self.codebase.codebase_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        # Also save pixi.toml if present.
        pixi_src = self.codebase.codebase_dir / "pixi.toml"
        if pixi_src.exists():
            shutil.copy2(pixi_src, snapshot_dir / "pixi.toml")
        # Record deleted files for diff computation.
        if parent_deleted:
            (snapshot_dir / ".deleted_files").write_text(
                "\n".join(parent_deleted) + "\n", encoding="utf-8",
            )
        _run_logger.info(f"[Node Snapshot] Saved for node {node.id[:8]}")
        return target_name

    def _find_node_snapshot(self, node: Any) -> str | None:
        """Find the snapshot directory for a tree node.

        Node snapshots are named iter_snapshot_<node.id[:8]>.

        Args:
            node: The SearchNode to look up.

        Returns:
            Path string if found, None otherwise.
        """
        target_name = f"iter_snapshot_{node.id[:8]}"
        snap_dir = self.codebase.codebase_dir / ".evorun_snapshots" / target_name
        if snap_dir.exists():
            return str(snap_dir)
        return None

    def _read_snapshot_code(self, node_or_snap_dir: Any, *, snap_dir: str | None = None) -> str:
        """Read all Python code from a node's snapshot directory.

        Args:
            node_or_snap_dir: A SearchNode (uses its snapshot) or a snapshot dir path string.
            snap_dir: Explicit snapshot directory path (overrides node_or_snap_dir).

        Returns:
            Concatenated Python code from all .py files in the snapshot,
            or empty string if snapshot not found.
        """
        if snap_dir:
            snap_dir_path = snap_dir
        elif isinstance(node_or_snap_dir, str):
            snap_dir_path = node_or_snap_dir
        else:
            snap_dir_path = self._find_node_snapshot(node_or_snap_dir)
        if not snap_dir_path:
            return ""
        parts: list[str] = []
        for py_file in sorted(Path(snap_dir_path).rglob("*.py")):
            try:
                code = py_file.read_text(encoding="utf-8")
                rel = py_file.relative_to(Path(snap_dir_path))
                parts.append(f"# File: {rel}\n{code}")
            except Exception:
                pass
        return "\n".join(parts)

    def _compute_diff_from_snapshot(
        self, snapshot_dir: str, modified_files: list[str],
        added_files: list[str], deleted_files: list[str],
    ) -> str:
        """Compute unified diff between snapshot files and current files.

        Handles modified, added, and deleted files. For added files the
        diff shows the full file as an addition; for deleted files the
        diff shows the full file as a deletion.

        Args:
            snapshot_dir: Path to the parent node's snapshot directory.
            modified_files: List of relative file paths that were modified.
            added_files: List of relative file paths that are new.
            deleted_files: List of relative file paths that were removed.

        Returns:
            Unified diff text (empty string if no diff or snapshot not found).
        """
        import difflib
        from pathlib import Path

        snap_path = Path(snapshot_dir)
        diffs: list[str] = []

        # --- Modified files ---
        for rel_file in modified_files:
            snap_file = snap_path / "experiment" / rel_file
            curr_file = self.codebase.codebase_dir / "experiment" / rel_file

            if not snap_file.exists() or not curr_file.exists():
                continue

            with open(snap_file) as f:
                old_lines = f.readlines()
            with open(curr_file) as f:
                new_lines = f.readlines()

            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"snapshot/{rel_file}",
                tofile=f"current/{rel_file}",
            )
            diffs.append("".join(diff))

        # --- Added files (full content as addition) ---
        for rel_file in added_files:
            curr_file = self.codebase.codebase_dir / "experiment" / rel_file
            if not curr_file.exists():
                continue
            with open(curr_file) as f:
                new_lines = f.readlines()
            diff = difflib.unified_diff(
                [], new_lines,
                fromfile=f"/dev/null",
                tofile=f"current/{rel_file}",
            )
            diffs.append("".join(diff))

        # --- Deleted files (full content as deletion) ---
        for rel_file in deleted_files:
            snap_file = snap_path / "experiment" / rel_file
            if not snap_file.exists():
                continue
            with open(snap_file) as f:
                old_lines = f.readlines()
            diff = difflib.unified_diff(
                old_lines, [],
                fromfile=f"snapshot/{rel_file}",
                tofile=f"/dev/null",
            )
            diffs.append("".join(diff))

        return "\n".join(diffs) if diffs else ""

    def _submit_edit_summary(
        self,
        diff_text: str,
        iteration: int,
    ) -> None:
        """Submit edit summary generation to background thread pool.

        Args:
            diff_text: The unified diff text to summarise.
            iteration: Current iteration number (for logging).
        """
        if self._summary_future is not None:
            # Cancel any pending summary from previous iteration.
            self._summary_future.cancel()

        def _do_summary():
            try:
                prompt = (
                    "You are a code-change summariser. Output exactly one line.\n\n"
                    "Summarise the following code diff in exactly ONE line. "
                    "Focus on what was changed (files, functions, strategies). "
                    f"Do NOT include code snippets. Keep it under {MAX_EDIT_SUMMARY_CHARS} characters.\n\n"
                    f"Diff:\n{diff_text}\n\nSummary:"
                )
                editor_env = _build_claude_env(self.editor_provider, self.editor_model, self.editor_api_key)
                summary = _run_claude_cli_with_env(
                    prompt,
                    cwd=str(self.codebase.codebase_dir),
                    model=self.editor_model,
                    env_overrides=editor_env,
                    max_turns=10,
                )
                return summary.strip()
            except Exception as e:
                _run_logger.warning(
                    f"Summarisation failed (iter {iteration}): {e}",
                )
                return "[summariser failed]"

        self._summary_future = self._summary_executor.submit(_do_summary)

    def _get_edit_summary(self, iteration: int) -> str:
        """Wait for and return the edit summary from the background thread.

        Args:
            iteration: Current iteration number (for logging).

        Returns:
            The edit summary string, or empty string if no summary was pending.
        """
        if self._summary_future is None:
            return ""

        try:
            summary = self._summary_future.result(timeout=600.0)
            return summary or ""
        except concurrent.futures.TimeoutError:
            _run_logger.warning(
                f"Summariser timed out (iter {iteration})"
            )
            return "[summariser timed out]"
        except Exception as e:
            _run_logger.warning(
                f"Summariser error (iter {iteration}): {e}"
            )
            return "[summariser error]"
        finally:
            self._summary_future = None

    def _restore_snapshot(self, snapshot_name: str) -> None:
        """Restore the codebase from a named snapshot.

        Restores all files copied during snapshot creation (experiment/ +
        pixi.toml).  The snapshot name can be a bare name such as
        ``"iter_snapshot_3"`` or an absolute directory path.  Absolute
        paths are validated to be inside the codebase's snapshot directory
        to prevent path-traversal attacks.

        Args:
            snapshot_name: Snapshot directory name or absolute path.
        """
        snapshot_dir = Path(snapshot_name)

        # Resolve relative names against the codebase snapshots directory.
        if not snapshot_dir.exists():
            snapshot_dir = (
                self.codebase.codebase_dir
                / ".evorun_snapshots"
                / snapshot_name
            )

        if not snapshot_dir.exists():
            _run_logger.info(f"[Restore] Snapshot not found: {snapshot_name}")
            return

        # Security: reject any resolved path that escapes the snapshot dir.
        try:
            resolved = snapshot_dir.resolve()
            allowed_base = (
                self.codebase.codebase_dir / ".evorun_snapshots"
            ).resolve()
            if not str(resolved).startswith(str(allowed_base)):
                _run_logger.error(
                    f"[Restore] Snapshot path escapes codebase: {snapshot_name}"
                )
                return
        except OSError:
            pass

        # Collect files that should exist (from the snapshot).
        snapshot_files: set[str] = set()
        for p in snapshot_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(snapshot_dir)
                snapshot_files.add(str(rel))
                target = self.codebase.codebase_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                _run_logger.info(f"  Restored: {rel}")

        # Remove stale files that exist in experiment/ but not in the snapshot.
        # Only scan the experiment/ subfolder — everything else (TASK.md,
        # eval.py, .gitignore, .git/, etc.) is left untouched.
        experiment_dir = self.codebase.codebase_dir / "experiment"
        if experiment_dir.is_dir():
            for p in experiment_dir.rglob("*"):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(self.codebase.codebase_dir))
                # Skip __pycache__ directories.
                if "__pycache__" in rel:
                    continue
                if rel not in snapshot_files:
                    try:
                        p.unlink()
                        _run_logger.info(f"  Removed stale: {rel}")
                    except OSError:
                        pass

# ─── Feedback formatting ─────────────────────────────────────

    def _format_feedback(
        self,
        result: EvalResult,
        iteration: int,
        tree_info: dict | None = None,
        prev_eval: str = "",
        code_review_notes: str = "",
        debug_instructions: str = "",
        parent_score: float | None = None,
    ) -> str:
        """Format evaluation feedback for Claude CLI.

        Orchestrates smaller formatting methods to build a natural-language
        message telling Claude about the latest evaluation results, what
        changed in the code, and what to focus on for the next round.

        Args:
            result: Latest evaluation results.
            iteration: Current iteration number.
            tree_info: Optional dict with tree search stats.
            prev_eval: Previous eval output from this node's last expansion.
            code_review_notes: Optional code review findings.
            debug_instructions: Optional debug agent instructions.

        Returns:
            Feedback message string (never None).
        """
        parts: list[str] = [f"### Iteration {iteration}/{self.max_iters}"]

        # --- Debug instructions (if available) ---
        if debug_instructions:
            parts.extend([debug_instructions, ""])

        # --- Code review notes (if available) ---
        if code_review_notes:
            parts.extend([code_review_notes, ""])

        # --- Clear improvement directive ---
        directive_lines = self._build_improvement_directive(result, iteration, parent_score)
        if directive_lines:
            parts.extend(directive_lines)
            parts.append("")

        # --- Scoring breakdown (included when score + description exist) ---
        if result.score is not None and result.description:
            parsed_metrics = self._parse_eval_description(result.description)
            if parsed_metrics:
                breakdown_lines = self._build_score_breakdown(parsed_metrics)
                if breakdown_lines:
                    parts.append("## Scoring breakdown")
                    parts.extend(breakdown_lines)
                    suggestions = self._build_improvement_suggestions(
                        parsed_metrics, result.description,
                    )
                    if suggestions:
                        parts.append("## Suggested improvements")
                        parts.extend(suggestions)

        # --- File changes from this iteration ---
        file_changes = self._format_file_changes()
        if file_changes:
            parts.extend(file_changes)

        # --- Task-specific hints ---
        task_tips = self._build_task_context(self.codebase._task_content)
        if task_tips:
            parts.append("## From TASK.md")
            parts.extend(task_tips)

        # --- Tree search feedback ---
        tree_lines = self._format_tree_context(tree_info)
        if tree_lines:
            parts.extend(tree_lines)

        # --- Previous eval output for debugging ---
        prev_eval_lines = self._format_prev_eval(prev_eval)
        if prev_eval_lines:
            parts.extend(prev_eval_lines)

        # --- Result info (timeout / score / error / speed warnings) ---
        result_lines = self._format_result_info(result)
        if result_lines:
            parts.extend(result_lines)

        # --- History summary ---
        history_lines = self._format_history_summary()
        if history_lines:
            parts.append("### Recent history\n" + "\n".join(history_lines))

        return "\n".join(parts)

    # ─── Fusion agent ───────────────────────────────────────────────

    def _find_fusion_candidates(self, target_node, max_candidates=3):
        """Find cross-branch fusion candidates from the search tree.

        Scans all nodes in the tree for high-scoring nodes from different
        branches than the target. Falls back to same-branch candidates
        if no cross-branch candidates exist.

        Args:
            target_node: The SearchNode being expanded.
            max_candidates: Maximum number of candidates to return.

        Returns:
            List of SearchNode objects (the reference solutions to fuse from),
            sorted by score descending.
        """
        target_branch_id = getattr(target_node, 'branch_id', None)
        candidates = []
        cross_branch_scores: dict[int, float] = {}

        for node in self.tree.journal.nodes:
            if node.id == target_node.id:
                continue
            if node is self.tree.root:
                continue
            score = node.metric.value if node.metric else None
            if score is None:
                continue
            node_branch = getattr(node, 'branch_id', None)
            if node_branch is None:
                continue
            if self.tree.maximize:
                if node_branch not in cross_branch_scores or score > cross_branch_scores[node_branch]:
                    cross_branch_scores[node_branch] = score
            else:
                if node_branch not in cross_branch_scores or score < cross_branch_scores[node_branch]:
                    cross_branch_scores[node_branch] = score

        for bid, s in cross_branch_scores.items():
            if target_branch_id is not None and bid == target_branch_id:
                continue
            for node in self.tree.journal.nodes:
                if getattr(node, 'branch_id', None) == bid and node.metric and node.metric.value == s:
                    candidates.append(node)
                    break
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            for node in self.tree.journal.nodes:
                if node.id == target_node.id:
                    continue
                if node is self.tree.root:
                    continue
                if getattr(node, 'branch_id', None) == target_branch_id:
                    score = node.metric.value if node.metric else None
                    if score is not None:
                        candidates.append(node)
                        if len(candidates) >= max_candidates:
                            break

        candidates.sort(
            key=lambda n: n.metric.value if n.metric else 0,
            reverse=self.tree.maximize,
        )
        return candidates[:max_candidates]

    def _run_fusion(self, target_node) -> tuple[str | None, str, list[str], list[str], list[str], bool]:
        """Run cross-branch fusion: merge techniques from other branches into the current node.

        Finds fusion candidates from other branches, builds a prompt that asks
        the LLM to selectively incorporate techniques, then calls Claude.

        Args:
            target_node: The SearchNode being expanded.

        Returns:
  Tuple of (log_content, modified_files, added_files, deleted_files, used_fusion).
            None is returned as the first element (response_text) since Claude CLI output is in log_content.
        """
        candidates = self._find_fusion_candidates(target_node, max_candidates=3)

        # No candidates at all — can't do fusion or fallback
        if not candidates:
            _run_logger.info(f"[Fusion] No candidates found for node {target_node.id[:8]}")
            return None, "", [], [], [], False

        # Build reference trajectories
        reference_sections = []
        for i, ref_node in enumerate(candidates):
            ref_score = ref_node.metric.value if ref_node.metric else "N/A"
            ref_branch = getattr(ref_node, 'branch_id', '?')
            reference_sections.append(
                f"## Reference Solution {i + 1} (Branch {ref_branch}, Score: {ref_score})\n"
                f"```\n{ref_node.code}\n```\n"
            )

        # Build codebase context
        codebase_prompt = self.codebase.get_codebase_prompt(max_size=4000)

        _run_logger.info(f"[Fusion] Running fusion for node {target_node.id[:8]} "
                        f"with {len(candidates)} reference(s)")

        # Two-stage fusion: planner → editor
        modified_files: list[str] = []
        added_files: list[str] = []
        deleted_files: list[str] = []
        log_content: str = ""

        prev_hashes = self._compute_file_hashes(self.codebase.get_experiment_files())

        # Fusion planner
        fusion_feedback = f"""\
### Cross-Branch Fusion Context

You are a code planning architect for cross-branch fusion. Analyze the
reference solutions below and plan how to selectively incorporate useful
techniques into the current codebase.

IMPORTANT RULES:
1. Do NOT write any code. Do NOT use Read, Edit, or Write tools.
2. Describe changes in plain language — the editor will implement them.

## Current Codebase
{codebase_prompt}

## Reference Solutions
{''.join(reference_sections)}

Produce a concise plan describing what techniques to incorporate and how.
"""
        fusion_plan = ""
        try:
            fusion_planner_env = _build_claude_env(self.planner_provider, self.planner_model, self.planner_api_key)
            fusion_plan = _run_claude_cli_with_env(
                prompt=fusion_feedback,
                cwd=str(self.codebase.codebase_dir),
                model=self.planner_model,
                env_overrides=fusion_planner_env,
                max_turns=30,
                allowed_tools=[],
                retries=2,
                retry_base_delay=3.0,
            ).strip()
        except Exception as e:
            _run_logger.warning(f"[Fusion] Planner failed: {e}")
            fusion_plan = ""

        if fusion_plan and fusion_plan.strip():
            # Fusion editor
            fusion_editor_env = _build_claude_env(self.editor_provider, self.editor_model, self.editor_api_key)
            fusion_editor_prompt = f"""\
You are implementing a cross-branch fusion plan.

## Fusion Plan
{fusion_plan}

## Current Codebase
{codebase_prompt}

## Reference Solutions (for context while reading)
{''.join(reference_sections)}

## Instructions
1. Read the current files to understand the codebase.
2. Implement the fusion plan: selectively adopt techniques from the
   reference solutions that fit your current architecture.
3. Preserve what is working in your current solution.
4. Do NOT blindly combine everything — choose the most relevant technique(s).
5. Output complete runnable Python files, not just diffs.
6. Do not suggest EDA.

Focus on quality over quantity: one well-integrated technique is better than
a messy combination of several.
"""
            try:
                log_content = _run_claude_cli_with_env(
                    prompt=fusion_editor_prompt,
                    cwd=str(self.codebase.codebase_dir),
                    model=self.editor_model,
                    env_overrides=fusion_editor_env,
                    max_turns=500,
                    allowed_tools=['Read', 'Edit', 'Write'],
                    log_file=str(self.codebase.codebase_dir / ".evorun_claude.log"),
                    retries=2,
                    retry_base_delay=3.0,
                )
            except Exception as e:
                _run_logger.error(f"[Fusion] Editor CLI failed: {e}")
                log_content = ""
        else:
            _run_logger.warning("[Fusion] Planner produced empty plan — skipping fusion")

        # Detect file changes
        current_hashes = self._compute_file_hashes(self.codebase.get_experiment_files())
        for fpath, chash in current_hashes.items():
            if fpath not in prev_hashes:
                added_files.append(fpath)
        for fpath in prev_hashes:
            if fpath not in current_hashes:
                deleted_files.append(fpath)
        for fpath, chash in current_hashes.items():
            if fpath in prev_hashes and prev_hashes[fpath] != chash:
                modified_files.append(fpath)

        used_fusion = bool(log_content)
        if used_fusion:
            _run_logger.info(f"[Fusion] Fusion successful: {len(modified_files)} modified, "
                           f"{len(added_files)} added, {len(deleted_files)} deleted")
        else:
            _run_logger.info(f"[Fusion] Fusion produced no changes — falling back to normal Claude")

        return None, log_content, modified_files, added_files, deleted_files, used_fusion

    def _format_file_changes(self) -> list[str]:
        """Format file changes section if any files were modified/added/deleted.

        Returns:
            List of lines for the file changes section, or empty list.
        """
        if not hasattr(self, "_last_modified_files") or not self._last_modified_files:
            return []
        lines: list[str] = ["## File changes this iteration"]
        if self._last_modified_files:
            lines.append(f"- **Modified**: {', '.join(self._last_modified_files)}")
        if self._last_added_files:
            lines.append(f"- **Added**: {', '.join(self._last_added_files)}")
        if self._last_deleted_files:
            lines.append(f"- **Deleted**: {', '.join(self._last_deleted_files)}")
        lines.append("")
        lines.append("These files are now part of the codebase. You may modify "
                     "any of them in addition to the files listed in Your Task.")
        return lines

    def _format_tree_context(self, tree_info: dict | None) -> list[str]:
        """Format tree search context section.

        Args:
            tree_info: Dict with tree search stats.

        Returns:
            List of lines for tree context, or empty list if no tree_info.
        """
        if not tree_info:
            return []
        lines: list[str] = [
            "## Tree context",
            f"- **Nodes in tree**: {tree_info['total_nodes']}",
            f"- **Total expansions**: {tree_info['total_expansions']}",
            f"- **Best score in tree**: {tree_info['best_score']:.4f}" if tree_info.get('best_score') is not None else "- **Best score in tree**: N/A",
        ]
        if tree_info.get("branch_stats"):
            lines.append("## Per-branch summary")
            for bs in tree_info["branch_stats"]:
                best_s = (
                    f"{bs['best_score']:.4f}" if bs["best_score"] is not None
                    else "N/A"
                )
                lines.append(
                    f"- **Branch {bs['id']}**: best={best_s}, "
                    f"evals={bs['evals_run']}, visits={bs['total_visits']}, "
                    f"depth={bs['depth']}"
                )
        if tree_info.get("failed_branches"):
            lines.append(
                f"- **Failed branches** (all evals returned None): "
                f"{tree_info['failed_branches']}"
            )
        if tree_info.get("avg_per_score") is not None:
            lines.append(
                f"- **Normalized avg**: {tree_info['avg_per_score']:.4f}"
            )
        return lines

    def _format_prev_eval(self, prev_eval: str) -> list[str]:
        """Format previous eval output section for debugging.

        Args:
            prev_eval: Previous eval output from this node's last expansion.

        Returns:
            List of lines, or empty list if no prev_eval.
        """
        if not prev_eval:
            return []
        return [
            "## Previous eval output (from last expansion of this node)",
            f"```\n{prev_eval.strip()[-MAX_PREV_EVAL_CHARS:]}\n```",
            "### IMPORTANT: This is the eval output from the code in "
            "this tree node. Study the error carefully and adjust the "
            "code to fix it.",
        ]

    def _format_result_info(self, result: EvalResult) -> list[str]:
        """Format result information (timeout, score, error, speed warnings).

        Handles the if/elif chain for different result states.

        Args:
            result: Latest evaluation results.

        Returns:
            List of lines for result info, or empty list.
        """
        if result.timed_out:
            return self._format_timeout_info(result)
        elif result.score is not None:
            return self._format_score_result(result)
        elif result.had_error and result.description:
            return [
                "### Eval crashed with error",
                f"- **Error**: {result.description[:500]}",
            ]
        elif result.output:
            return self._format_eval_failure(result)
        elif (result.exec_time or 0.0) > self.eval_timeout * SPEED_CRITICAL_THRESHOLD:
            return self._format_critical_speed_warning(result)
        elif (result.exec_time or 0.0) > self.eval_timeout * SPEED_WARNING_THRESHOLD:
            return self._format_speed_warning(result)
        return []

    def _format_timeout_info(self, result: EvalResult) -> list[str]:
        """Format timeout information section.

        Args:
            result: Eval result that timed out.

        Returns:
            List of lines for timeout info.
        """
        et = result.exec_time if result.exec_time is not None else self.eval_timeout
        lines: list[str] = [
            f"### Eval TIMED OUT after {et:.0f}s",
            f"- **Timeout limit**: {self.eval_timeout}s",
            "- **Process killed by OS signal** (SIGKILL)",
            "",
            "## Captured stdout/stderr before kill:",
        ]
        if result.output:
            lines.append("```\n")
            for line in result.output[:MAX_OUTPUT_LINES]:
                lines.append(str(line)[:MAX_OUTPUT_LINE_LEN])
            lines.append("\n```")
        else:
            lines.append("None — nothing output before kill.")

        lines.extend([
            "",
            "## Why this matters",
            "A timeout means the eval command was killed by the OS before returning. "
            "Common causes:",
            "- **Infinite loop or unbounded recursion**",
            "- **Blocking I/O** — print(input()), sys.stdin.read()",
            "- **Expensive computation** — O(n^2) or worse",
            "- **Network call with no timeout**",
            "- **Blocking file I/O**",
            "",
            "## What to check first",
            "1. Review all loops (while/for) — do they have proper termination?",
            "2. Remove blocking I/O (no print(input()), no sys.stdin.read())",
            "3. Avoid expensive operations on large datasets",
            "4. Add early exits (break when done)",
            "5. Check if any data loading could be blocking",
            "",
            "## PRIORITY: The eval timed out — this is a hard fail.",
            "Every iteration that times out wastes time without making progress.",
            "You MUST make the code run faster. Do not try to improve scores if "
            "the eval command cannot complete.",
        ])
        return lines

    def _format_common_timeout_causes(self, include_output_clue: bool = False) -> list[str]:
        """Format common timeout causes and troubleshooting checklist.

        Args:
            include_output_clue: Whether to add the captured-output clue line.

        Returns:
            List of lines for the common causes section.
        """
        lines: list[str] = [
            "",
            "## Why this matters",
            "Common causes:",
            "- **Infinite loop or unbounded recursion** — check all while/for loops",
            "- **Blocking I/O** — print(input()), sys.stdin.read() can hang forever",
            "- **Expensive computation** — O(n^2) or worse on large datasets",
            "- **Network call with no timeout** — stuck waiting for a response",
            "",
            "## What to check",
            "1. Review all loops for proper termination conditions",
            "2. Remove any blocking I/O calls (print(input()), sys.stdin.read())",
            "3. Avoid expensive operations on large datasets",
            "4. Use bounded/early-exit loops",
        ]
        if include_output_clue:
            lines.append(
                "5. **The captured output above is what happened before the kill —"
            )
            lines.append(
                "   that is your strongest clue for what went wrong.**",
            )
        return lines

    def _format_score_result(self, result: EvalResult) -> list[str]:
        """Format successful score result.

        Args:
            result: Eval result with a valid score.

        Returns:
            List of lines with score, description, and execution time.
        """
        et = result.exec_time if result.exec_time is not None else 0.0
        return [
            f"**Score**: {result.score:.4f}",
            f"**Description**: {result.description}",
            f"**Execution Time**: {et:.1f}s / {self.eval_timeout}s",
        ]

    def _format_eval_failure(self, result: EvalResult) -> list[str]:
        """Format eval failure with raw output.

        Args:
            result: Eval result that failed without a score.

        Returns:
            List of lines with error and captured output.
        """
        lines: list[str] = ["## Eval failed (no JSON output)", ""]
        if result.description:
            lines.append(f"- **Error**: {result.description}")
            lines.append("")
        lines.append("### Captured output:")
        lines.append("```")
        for line in result.output[:MAX_OUTPUT_LINES]:
            lines.append(str(line)[:MAX_OUTPUT_LINE_LEN])
        lines.append("```\n")
        return lines

    def _format_critical_speed_warning(self, result: EvalResult) -> list[str]:
        """Format critical speed warning (90%+ of timeout).

        Args:
            result: Eval result approaching timeout.

        Returns:
            List of lines for critical speed warning.
        """
        lines: list[str] = [
            f"**CRITICAL**: Last run took {result.exec_time:.1f}s "
            f"(90% of {self.eval_timeout}s timeout)",
            "Minor regressions will cause timeouts.",
        ]
        if result.output:
            lines.append("```\n")
            for line in result.output[:MAX_OUTPUT_LINES]:
                lines.append(str(line)[:MAX_OUTPUT_LINE_LEN])
            lines.append("\n```")
        else:
            lines.append("None — nothing was output before the process was killed.")
        lines.extend(self._format_common_timeout_causes(include_output_clue=True))
        return lines

    def _format_speed_warning(self, result: EvalResult) -> list[str]:
        """Format speed warning (70-90% of timeout).

        Args:
            result: Eval result approaching timeout.

        Returns:
            List of lines for speed warning.
        """
        return [
            f"**WARNING**: Last run took {result.exec_time:.0f}s, "
            f"approaching the timeout of {self.eval_timeout}s.",
            "If the score isn't strong, try simplifying.",
        ]

    def _format_history_summary(self) -> list[str]:
        """Format recent history summary (last 5 iterations).

        Returns:
            List of history summary lines, or empty list if no history.
        """
        if not self.history:
            return []
        recent = self.history[-5:]
        history_lines: list[str] = []
        for entry in recent:
            et = entry.exec_time if entry.exec_time is not None else 0.0
            summary = entry.edit_summary
            file_parts: list[str] = []
            if entry.files_modified:
                file_parts.append(f"~{len(entry.files_modified)} modified")
            if entry.files_added:
                file_parts.append(f"+{len(entry.files_added)} added")
            if entry.files_deleted:
                file_parts.append(f"-{len(entry.files_deleted)} deleted")
            file_str = ", ".join(file_parts) if file_parts else ""
            ts = ""
            if entry.datetime:
                try:
                    ts = datetime.fromisoformat(entry.datetime).strftime("%H:%M:%S")
                except (ValueError, TypeError):
                    pass
            time_str = f" [{ts}]" if ts else ""
            if entry.timed_out:
                line = f"Iter {entry.iter}: timed out ({et:.0f}s){time_str}"
                if file_str: line += f" [{file_str}]"
                if summary: line += f" [{summary}]"
            elif entry.score is not None:
                line = f"Iter {entry.iter}: score={entry.score:.4f} ({et:.0f}s){time_str}"
                if file_str: line += f" [{file_str}]"
                if summary: line += f" [{summary}]"
            else:
                line = f"Iter {entry.iter}: no score ({et:.0f}s){time_str}"
                if file_str: line += f" [{file_str}]"
                if summary: line += f" [{summary}]"
            history_lines.append(line)
        return history_lines

    def _build_score_breakdown(
        self, scored_parts: list[tuple[str, float]]
    ) -> list[str]:
        """Build a human-readable score breakdown from parsed metrics.

        Args:
            scored_parts: List of (name, value) pairs from eval.description.

        Returns:
            Lines suitable for markdown formatting.
        """
        if not scored_parts:
            return []

        # Metric bounds with direction: 'high' = higher value is better,
        # 'low' = lower value is better.
        bounds_by_name: dict[str, dict[str, Any]] = {
            "score": {"lower": 0, "upper": 1, "direction": "high"},
            "accuracy": {"lower": 0, "upper": 1, "direction": "high"},
            "rmse": {"lower": 0, "upper": float("inf"), "direction": "low"},
            "mse": {"lower": 0, "upper": float("inf"), "direction": "low"},
            "mae": {"lower": 0, "upper": float("inf"), "direction": "low"},
            "loss": {"lower": 0, "upper": float("inf"), "direction": "low"},
            "precision": {"lower": 0, "upper": 1, "direction": "high"},
            "recall": {"lower": 0, "upper": 1, "direction": "high"},
            "f1": {"lower": 0, "upper": 1, "direction": "high"},
            "r2": {"lower": 0, "upper": 1, "direction": "high"},
            "log_loss": {"lower": 0, "upper": float("inf"), "direction": "low"},
            "auc": {"lower": 0, "upper": 1, "direction": "high"},
        }

        parts: list[str] = []
        for key, value in scored_parts:
            name = key.replace("_", " ").capitalize()
            if key in bounds_by_name:
                b = bounds_by_name[key]
                lower = b["lower"]
                upper = b["upper"]
                direction = b["direction"]  # 'high' or 'low' in absolute terms

                # Compute normalized score [0, 1] where 1 = best
                if upper == float("inf"):
                    # Error metric (RMSE, etc.): use 1/(1+sqrt(v)) so 0→1.0, large→0
                    norm = 1.0 / (1.0 + (value ** 0.5)) if value > 0 else 1.0
                else:
                    norm = max(0.0, min(1.0, (value - lower) / (upper - lower)))

                # If direction is 'low' (lower is better), invert the norm
                if direction == "low":
                    norm = 1.0 - norm

                # Invert again if minimization mode and direction is 'high'
                # (e.g. minimizing score: a high score value is "bad" in minimize mode)
                if not self.tree.maximize and direction == "high":
                    norm = 1.0 - norm

                if norm >= 0.8:
                    bar = "🟩" * min(10, int(norm * 10))
                elif norm >= 0.4:
                    bar = "🟨" * min(10, int(norm * 10))
                else:
                    bar = "⬜" * max(1, min(10, int(norm * 10)))
                parts.append(f"- {name}: {value:.4f} (bar: [{bar}])")
            else:
                parts.append(f"- {name}: {value:.4f}")
        return parts

    def _build_task_context(self, task_content: str | None) -> list[str]:
        """Build task-specific context from the loaded TASK.md file.

        Args:
            task_content: Contents of TASK.md or None if not found.

        Returns:
            List of context lines for markdown formatting.
        """
        import evorun.utils.response as _utils
        if not task_content:
            return []

        parts: list[str] = []
        try:
            task_data = _utils.extract_jsons(task_content)
            if task_data:
                for item in task_data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    parts.append(
                                        f"- {key}: {k} = {v}"
                                    )
                            else:
                                parts.append(f"- {key}: {value}")
        except (ValueError, AttributeError):
            # Fallback to plain text
            for line in task_content.splitlines():
                if line.strip():
                    parts.append(f"- {line.strip()}")
        return parts

    def _build_improvement_directive(
        self, result: EvalResult, iteration: int, parent_score: float | None = None,
    ) -> list[str]:
        """Build a clear, actionable improvement directive for Claude.

        Args:
            result: Latest evaluation results.
            iteration: Current iteration number.

        Returns:
            List of directive lines for markdown formatting.
        """
        if result.score is None:
            return []

        # Try to extract target score from TASK.md
        target_score = None
        if self.codebase._task_content:
            for line in self.codebase._task_content.splitlines():
                if any(kw in line.lower() for kw in ["score >=", "target score", "try to reach"]):
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        target_score = float(match.group(1))
                        break

        directive: list[str] = ["## Your Task"]
        if target_score is not None:
            improved_target = (result.score < target_score) if self.tree.maximize else (result.score > target_score)
            if improved_target:
                op = ">=" if self.tree.maximize else "<="
                directive.append(
                    f"**Improve the code to reach score {op} {target_score}**"
                )
                directive.append(
                    f"Current score: {result.score:.4f} (not yet at target)"
                )
            else:
                directive.append(
                    f"**You've exceeded the target ({target_score})!**"
                )
                directive.append(
                    f"Current score: {result.score:.4f}. Continue improving if possible."
                )
        else:
            direction = "increase the evaluation score" if self.tree.maximize else "decrease the evaluation score"
            directive.append(f"**Improve the code to {direction}**")
            directive.append(f"Current score: {result.score:.4f}")

        if parent_score is not None:
            delta = result.score - parent_score
            improved = (delta > 0) if self.tree.maximize else (delta < 0)
            direction = "improved" if improved else "decreased"
            directive.append(
                f"Previous score: {parent_score:.4f} -> Current: {result.score:.4f} ({delta:+.4f})"
            )

        directive.append("")
        directive.append("You may modify these files:")

        # List modifiable files from TASK.md
        if self.codebase._task_content:
            mod_files = []
            in_modifiable = False
            for line in self.codebase._task_content.splitlines():
                if "you may only modify" in line.lower():
                    in_modifiable = True
                    continue
                if in_modifiable:
                    if line.strip().startswith("-"):
                        mod_files.append(line.strip()[1:].strip())
                    elif line.strip().startswith("#"):
                        break
            if mod_files:
                for f in mod_files:
                    directive.append(f"- {f}")
            else:
                # Fallback: list actual experiment files.
                exp_files = self.codebase.get_experiment_files()
                for f in exp_files:
                    directive.append(f"- {f}")
        else:
            # No TASK.md — list actual experiment files.
            exp_files = self.codebase.get_experiment_files()
            for f in exp_files:
                directive.append(f"- {f}")

        directive.append("\nYou may NOT modify eval.py, evaluation.py, or external test harnesses.\n")
        directive.append("**Make specific code changes now to improve the score.**")

        # Add performance-based hints
        if result.description:
            hint_lines = self._build_metric_hints(result)
            if hint_lines:
                directive.append("")
                directive.append("Performance notes:")
                directive.extend(hint_lines)

        # Add relevant task hints
        hints = self._extract_task_hints(self.codebase._task_content)
        if hints:
            directive.append("")
            directive.append("Relevant strategies:")
            for hint in hints[:3]:
                directive.append(f"- {hint}")

        return directive

    def _build_metric_hints(self, result: EvalResult) -> list[str]:
        """Generate specific hints based on eval metrics from the description.

        Args:
            result: Latest evaluation results.

        Returns:
            List of hint strings.
        """
        hints: list[str] = []
        desc = result.description.lower()

        mse_match = re.search(r'mse[=:\s]+([0-9]+\.?\d*)', desc)
        if mse_match:
            mse = float(mse_match.group(1))
            if self.tree.maximize and mse > 1.0:
                hints.append(f"- **MSE ({mse:.6f}) is high** — focus on reducing prediction error")
            elif not self.tree.maximize:
                hints.append(f"- MSE: {mse:.6f} (your target is the overall score)")

        speed_match = re.search(r'(\d+\.?\d*)\s*(?:ms|milliseconds?)', desc)
        if speed_match:
            time_ms = float(speed_match.group(1))
            if time_ms > 20:
                hints.append(f"- **Speed ({time_ms:.1f}ms) is slow** - optimize for faster execution")
            elif time_ms > 10:
                hints.append("- Speed could be improved (current: {:.1f}ms)".format(time_ms))

        acc_match = re.search(r'acc(?:uracy)?(?:_?factor)?[=:\s]+([0-9]+\.?\d*)', desc)
        if acc_match:
            acc = float(acc_match.group(1))
            if self.tree.maximize:
                if acc < 0.5:
                    hints.append(f"- Accuracy factor ({acc:.3f}) is low")
                elif acc < 0.8:
                    hints.append(f"- Accuracy factor could improve ({acc:.3f})")
            else:
                hints.append(f"- Accuracy factor: {acc:.3f} (your target is the overall score)")

        return hints

    def _build_improvement_suggestions(
        self,
        scored_parts: list[tuple[str, float]],
        desc: str,
    ) -> list[str]:
        """Compute improvement suggestions from weak metrics in the scored parts.

        Args:
            scored_parts: List of (name, value) pairs from eval description.
            desc: The full description string (for fallback parsing).

        Returns:
            List of suggestion strings for markdown formatting.
        """
        parts: list[str] = []

        # Parse out numeric metrics.
        metrics: dict[str, float] = {}
        for key, value in scored_parts:
            metrics[key] = value

        # Check if any metrics exist.
        if not metrics:
            return parts

       # Build suggestions based on metric scores.
        # In maximization mode, absolute thresholds are appropriate
        # (low accuracy = bad, high RMSE = bad). In minimization mode,
        # sub-metrics contribute to the overall score differently,
        # so we report values without directional advice.
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            if self.tree.maximize and acc < 0.5:
                parts.append(
                    "- Accuracy is low (<0.5). Consider a different model or "
                    "architecture."
                )
            elif self.tree.maximize and acc < 0.8:
                parts.append(
                    f"- Accuracy could be improved (current: {acc:.4f}). "
                    "Try tuning hyperparameters."
                )
            elif not self.tree.maximize:
                parts.append(f"- Accuracy: {acc:.4f} (note: your target is the overall score)")

        if "rmse" in metrics:
            rmse = metrics["rmse"]
            if self.tree.maximize and rmse > 1.0:
                parts.append(
                    f"- High RMSE ({rmse:.4f}). Review model performance."
                )
            elif not self.tree.maximize:
                parts.append(f"- RMSE: {rmse:.4f} (note: your target is the overall score)")

        if "fit_time_ms" in metrics:
            fit_time = metrics["fit_time_ms"]
            if fit_time > 5000:
                parts.append(f"- Execution time ({fit_time:.2f}ms) could be "
                                "improved.")

        return parts

    def _compute_file_hashes(self, files: list[str]) -> dict[str, str]: 
        """Compute SHA-256 hashes of Python files in the codebase.

        Used to detect meaningful changes made by the LLM.

        Args:
            files: List of relative file paths (relative to experiment/).

        Returns:
            Dict mapping file path to its SHA-256 hash.
        """
        hashes: dict[str, str] = {}
        for rel_path in files:
            file_path = self.codebase.codebase_dir / "experiment" / rel_path
            if file_path.exists():
                content = file_path.read_bytes()
                hashes[rel_path] = hashlib.sha256(content).hexdigest()
        return hashes


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────


# Default values for argparse arguments (used to detect if CLI was explicitly set).
_ARGPARSE_DEFAULTS: dict[str, Any] = {
    "max_children": 10,
    "optim_mode": "max",
    "max_iters": 50,
    "time_limit": None,
    "patience": 10,
    "eval_timeout": 300,
    "llm_retries": 3,
    "llm_retry_base_delay": 3.0,
    "decay_exploration": True,
    "fake_run": False,
    "reset": False,
    "eval_cmd": None,
    "use_fusion": True,
    "fusion_min_iters": 10,
    "fusion_prob": 0.5,
}

# Mapping from config.toml keys to argparse destination names.
_CONFIG_TO_ARGPARSE: dict[str, str] = {
    "eval_cmd": "eval_cmd",
    "optim_mode": "optim_mode",
    "max_iters": "max_iters",
    "time_limit": "time_limit",
    "patience": "patience",
    "eval_timeout": "eval_timeout",
    "max_children": "max_children",
    "llm_retries": "llm_retries",
    "llm_retry_base_delay": "llm_retry_base_delay",
    "decay_exploration": "decay_exploration",
    "fake_run": "fake_run",
    "reset": "reset",
    "use_fusion": "use_fusion",
    "fusion_min_iters": "fusion_min_iters",
    "fusion_prob": "fusion_prob",
}


def _load_task_config(codebase_dir: Path) -> dict[str, Any]:
    """Load task-specific configuration from config.toml.

    Reads <codebase_dir>/config.toml and returns a dict of configuration values.
    Uses tomllib (Python 3.11+) or tomli as fallback.

    Args:
        codebase_dir: Path to the codebase directory.

    Returns:
        Dict of configuration values, or empty dict if file is missing/invalid.
    """
    config_path = codebase_dir / "config.toml"
    if not config_path.exists():
        return {}
    try:
        # Try tomllib first (Python 3.11+), fall back to tomli.
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(config_path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        return data or {}
    except Exception as e:
        _run_logger.warning(f"Failed to load {config_path}: {e}")
        return {}


def _apply_task_config(args: argparse.Namespace, task_config: dict[str, Any]) -> None:
    """Override unset CLI args with values from config.toml.

    Only overrides a value if the CLI argument equals its argparse default
    (meaning the user didn't explicitly set it on the command line).

    Args:
        args: Parsed argparse namespace.
        task_config: Configuration dict loaded from config.toml.
    """
    for config_key, arg_dest in _CONFIG_TO_ARGPARSE.items():
        if config_key not in task_config:
            continue

        cli_val = getattr(args, arg_dest, None)
        default_val = _ARGPARSE_DEFAULTS.get(arg_dest)

        # Only override if CLI value matches the argparse default.
        if cli_val == default_val:
            value = task_config[config_key]

            setattr(args, arg_dest, value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments using argparse.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Iteratively improve a codebase using LLM-driven optimization. "
        "With Monte Carlo Tree Search for branching exploration.",
    )

    # Codebase directory (defaults to cwd).
    parser.add_argument(
        "codebase_dir", nargs="?", default=".",
        help="Directory containing the codebase experiment (default: current directory)",
    )

    # Eval command.
    parser.add_argument(
        "--eval-cmd",
        default=None,
        help="Evaluation command to run (default: 'pixi run python eval.py')",
    )

    # Optional arguments.
    parser.add_argument(
        "--max-children",
        type=int, default=10,
        help="Maximum child expansions per node in the tree (default: 10)"
    )
    parser.add_argument(
        "--optim-mode",
        choices=["max", "min"],
        default="max",
        help="Optimization direction: 'max' to maximize score, 'min' to minimize (default: max)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing state file and start fresh",
    )
    parser.add_argument(
        "--model", "-m", default="claude-sonnet-4-6",
        help="Claude model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-iters", type=int, default=50,
        help="Maximum number of iterations (default: 50)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=None,
        help="Wall-clock time limit in seconds (no limit if not specified)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Stagnation threshold: stop when all branches have been stagnant "
              "for >=N iterations (default: 10)",
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=300,
        help="Timeout per evaluation run in seconds (default: 300)",
    )
    parser.add_argument(
        "--llm-retries", type=int, default=3,
        help="Retry LLM query on failure (default: 3)",
    )
    parser.add_argument(
        "--llm-retry-base-delay", type=float, default=3.0,
        help="Base delay (s) for exponential backoff (default: 3.0)",
    )
    parser.add_argument(
        "--fake-run",
        action="store_true",
        help=("Run in 'fake' mode: generate random scores, simulate LLM "
              "changes — for rapid tree-search prototyping without "
              "real eval or LLM calls"),
    )
    parser.add_argument(
        "--decay-exploration",
        action="store_true",
        default=True,
        help="Enable progressive UCT exploration decay over time (default: True)",
    )
    parser.add_argument(
        "--no-decay",
        action="store_false",
        dest="decay_exploration",
        help="Disable progressive UCT exploration decay",
    )
    parser.add_argument(
        "--no-fusion",
        action="store_false",
        dest="use_fusion",
        help="Disable cross-branch fusion agent (default: enabled)",
    )
    parser.add_argument(
        "--fusion-min-iters",
        type=int,
        default=10,
        help="Minimum iterations before fusion is allowed (default: 10)",
    )
    parser.add_argument(
        "--fusion-prob",
        type=float,
        default=0.5,
        help="Probability of using fusion after fusion_min_iters (default: 0.5)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the web visualization server alongside the evorun run",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Start the web visualization server without running the optimizer (blocks until interrupted)",
    )
    parser.add_argument(
        "--port",
        type=int, default=9000,
        help="Port for the web visualization server (default: 9000)",
    )

    args = parser.parse_args()

    # Load task-specific config from config.toml (precedence: CLI > config.toml > config.yaml > defaults).
    task_config = _load_task_config(Path(args.codebase_dir))
    if task_config:
        _apply_task_config(args, task_config)
        _run_logger.info(f"Loaded config.toml from {args.codebase_dir}")

    # Validate arguments.
    if not Path(args.codebase_dir).exists():
        raise FileNotFoundError(f"codebase_dir not found: {args.codebase_dir}")
    if args.max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if args.patience < 1:
        raise ValueError("patience must be >= 1")
    if args.eval_timeout < 1:
        raise ValueError("eval_timeout must be >= 1")

    # Validate required folder structure (skip for server-only mode).
    if not getattr(args, 'server_only', False):
        codebase_root = Path(args.codebase_dir)
        if not (codebase_root / "experiment").is_dir():
            raise FileNotFoundError(
                f"Required 'experiment/' folder not found in: {codebase_root}"
            )
        if not (codebase_root / "TASK.md").is_file():
            raise FileNotFoundError(
                f"Required 'TASK.md' file not found in: {codebase_root}"
            )

    if not args.reset and Path(args.codebase_dir).exists():
        state_path = Path(args.codebase_dir) / ".evorun_state.json"
        if state_path.exists():
            _run_logger.info(f"Found state file: {state_path} — resuming.")

    return args


# --------------------
# Utilities
# --------------------


def _setup_logging() -> None:
    """Configure logging for evorun.py."""
    log = logging.getLogger("evorun")
    if log.handlers:
        return  # Already configured
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.addHandler(handler)


def _split_lines(text: str) -> list[str]:
    """Split stdout/stderr into cleaned lines, deduplicated."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


def main() -> None:
    """Entry point for evorun.py.

    Validates inputs, parses CLI arguments, creates an instance of
    EvoRunAgent, and runs the optimization loop.
    """
    _setup_logging()
    args = parse_args()

    # --reset: delete all .evorun artifacts before starting.
    if getattr(args, 'reset', False) and args.codebase_dir:
        codebase_root = Path(args.codebase_dir)
        _evorun_artifacts = [
            ".evorun_state.json",
            ".evorun_backup",
            ".evorun_snapshots",
            ".evorun_claude.log",
        ]
        for name in _evorun_artifacts:
            p = codebase_root / name
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                    _run_logger.info(f"[Reset] Deleted directory: {p}")
                else:
                    p.unlink()
                    _run_logger.info(f"[Reset] Deleted file: {p}")

    # Optionally start the web visualization server in a background thread.
    _server = None
    if getattr(args, 'server_only', False):
        from .webapp.server import start_server
        _server = start_server(
            folder=str(args.codebase_dir),
            port=args.port,
            open_browser=True,
        )
        _run_logger.info(
            f"[Server] Visualizer started on http://localhost:{args.port} (server-only mode)"
        )
        _server_thread = threading.Thread(
            target=_server.serve_forever, daemon=True
        )
        _server_thread.start()
        try:
            _server_thread.join()
        except KeyboardInterrupt:
            _run_logger.info("[KeyboardInterrupt] User cancelled.")
        finally:
            _server.shutdown()
            _run_logger.info("[Server] Visualizer stopped.")
        sys.exit(0)

    if getattr(args, 'server', False):
        from .webapp.server import start_server
        _server = start_server(
            folder=str(args.codebase_dir),
            port=args.port,
            open_browser=True,
        )
        _run_logger.info(
            f"[Server] Visualizer started on http://localhost:{args.port}"
        )
        # Detach the server thread so it doesn't block main().
        _server_thread = threading.Thread(
            target=_server.serve_forever, daemon=True
        )
        _server_thread.start()

    agent = None
    try:
        agent = EvoRunAgent(args)
        agent.run()
    except KeyboardInterrupt:
        _run_logger.info("[KeyboardInterrupt] User cancelled.")
        sys.exit(1)
    except Exception:
        _run_logger.exception("[Error] Unexpected exception in EvoRunAgent")
        sys.exit(2)
    finally:
        # Shut down the background thread pool.
        if agent is not None and hasattr(agent, '_summary_executor'):
            agent._summary_executor.shutdown(wait=False)
        # Stop the visualizer server.
        if _server is not None:
            _server.shutdown()
            _run_logger.info("[Server] Visualizer stopped.")



if __name__ == "__main__":
    main()
