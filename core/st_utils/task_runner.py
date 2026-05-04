"""
Background task runner for Streamlit with pause/resume/stop control.

Usage:
    runner = TaskRunner.get(st.session_state)
    runner.start(steps)  # list of (label, callable) tuples
    runner.pause() / runner.resume() / runner.stop()
    runner.state  # "idle" | "running" | "paused" | "stopped" | "completed" | "error"
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


class StopTask(Exception):
    """Raised when the task is stopped by user."""

    pass


@dataclass
class TaskRunner:
    """Manages a background thread that executes a sequence of steps with pause/stop control."""

    # Public read-only state
    state: str = "idle"  # idle | running | paused | stopped | completed | error
    current_step: int = -1  # 0-indexed, -1 = not started
    total_steps: int = 0
    current_label: str = ""
    error_msg: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    current_step_started_at: float | None = None
    step_durations: list[tuple[str, float]] = field(default_factory=list)

    # Internal
    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list = field(default_factory=list)

    def __post_init__(self):
        self._pause_event.set()  # not paused initially

    # ------ Singleton per session_state ------
    @staticmethod
    def get(session_state, key: str = "_task_runner") -> "TaskRunner":
        """Get or create a TaskRunner stored in Streamlit session_state."""
        if key not in session_state:
            session_state[key] = TaskRunner()
        return session_state[key]

    # ------ Control API ------

    def start(self, steps: list[tuple[str, Callable]]):
        """Start executing steps in a background thread.

        Args:
            steps: list of (label, callable) — each callable takes no args.
        """
        if self.state == "running" or self.state == "paused":
            return  # already running

        self._steps = steps
        self.total_steps = len(steps)
        self.current_step = -1
        self.current_label = ""
        self.error_msg = ""
        self.started_at = time.perf_counter()
        self.finished_at = None
        self.current_step_started_at = None
        self.step_durations = []
        self.state = "running"

        self._pause_event.set()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        if self.state == "running":
            self._pause_event.clear()
            self.state = "paused"

    def resume(self):
        if self.state == "paused":
            self._pause_event.set()
            self.state = "running"

    def stop(self):
        """Request stop. The task will halt before the next step."""
        if self.state in ("running", "paused"):
            self._stop_event.set()
            self._pause_event.set()  # unblock if paused so thread can exit
            self.state = "stopped"

    def reset(self):
        """Reset to idle state (only when not running)."""
        if self.state not in ("running", "paused"):
            self.state = "idle"
            self.current_step = -1
            self.total_steps = 0
            self.current_label = ""
            self.error_msg = ""
            self.started_at = None
            self.finished_at = None
            self.current_step_started_at = None
            self.step_durations = []
            self._steps = []

    @property
    def is_active(self) -> bool:
        return self.state in ("running", "paused")

    @property
    def is_done(self) -> bool:
        return self.state in ("completed", "stopped", "error")

    @property
    def progress(self) -> float:
        """0.0 to 1.0"""
        if self.total_steps == 0:
            return 0.0
        return min((self.current_step + 1) / self.total_steps, 1.0)

    @property
    def total_elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end_time = (
            self.finished_at
            if self.finished_at is not None
            else time.perf_counter()
        )
        return max(0.0, end_time - self.started_at)

    @property
    def current_step_elapsed(self) -> float:
        if self.current_step_started_at is None or not self.is_active:
            return 0.0
        return max(0.0, time.perf_counter() - self.current_step_started_at)

    # ------ Internal ------

    def _run(self):
        """Execute steps sequentially in background thread."""
        try:
            for i, (label, func) in enumerate(self._steps):
                # Check stop before each step
                if self._stop_event.is_set():
                    self.state = "stopped"
                    self.finished_at = time.perf_counter()
                    return

                # Block if paused
                self._pause_event.wait()

                # Check stop again after resume
                if self._stop_event.is_set():
                    self.state = "stopped"
                    self.finished_at = time.perf_counter()
                    return

                self.current_step = i
                self.current_label = label
                self.current_step_started_at = time.perf_counter()
                func()
                step_elapsed = time.perf_counter() - self.current_step_started_at
                self.step_durations.append((label, step_elapsed))
                self.current_step_started_at = None

            self.state = "completed"
            self.finished_at = time.perf_counter()
        except Exception as e:
            if self.current_step_started_at is not None and self.current_label:
                step_elapsed = time.perf_counter() - self.current_step_started_at
                self.step_durations.append((self.current_label, step_elapsed))
                self.current_step_started_at = None
            self.error_msg = str(e)
            self.finished_at = time.perf_counter()
            self.state = "error"
