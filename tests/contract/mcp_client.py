"""A synchronous JSON-RPC client for driving an MCP server over stdio.

Contract tests talk to a real server process, and MCP's own async client brings
anyio cancel scopes that cannot be exited from a different task than they were
entered in - which pytest-asyncio fixtures do routinely. Driving the pipes
synchronously avoids that entirely and matches how OpenCode actually runs the
server: a subprocess, newline-delimited JSON on stdin/stdout.

Reads run on a background thread so a dead server fails with a timeout instead
of blocking forever.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

DEFAULT_TIMEOUT = 120.0


@dataclass
class ToolResult:
    """The outcome of an MCP tool call."""

    is_error: bool
    text: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ToolResult:
        blocks = payload.get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return cls(is_error=bool(payload.get("isError")), text=text)

    def json(self) -> Any:
        """Decode the tool's JSON response body."""
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Tool did not return JSON: {self.text[:400]!r}") from e


class McpStdioClient:
    """Spawns an MCP server subprocess and speaks JSON-RPC to it."""

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self._next_id = 0
        self._responses: Queue[dict[str, Any]] = Queue()
        self._stderr: list[str] = []

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env or dict(os.environ),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        self._readers = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    # ==================== Background pumping ====================

    def _pump_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                # Servers should only emit JSON-RPC; ignore stray output rather
                # than wedging the reader thread.
                continue

    def _pump_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    # ==================== Protocol ====================

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP handshake and return the server's result."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "contract-tests", "version": "1.0"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        return ToolResult.from_payload(
            self.request("tools/call", {"name": name, "arguments": arguments or {}})
        )

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return its result, raising on error or timeout."""
        self._next_id += 1
        request_id = self._next_id

        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                message = self._responses.get(timeout=0.5)
            except Empty:
                if self._process.poll() is not None:
                    raise AssertionError(
                        f"Server exited with code {self._process.returncode}\n{self.stderr_tail()}"
                    ) from None
                continue

            if message.get("id") != request_id:
                continue  # a notification, or a reply to another request

            if "error" in message:
                raise AssertionError(f"Server returned an error: {message['error']}")

            return message.get("result", {})

        raise TimeoutError(
            f"No response to {method!r} within {self.timeout}s\n{self.stderr_tail()}"
        )

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification, which expects no reply."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        if self._process.poll() is not None:
            raise AssertionError(
                f"Server exited with code {self._process.returncode}\n{self.stderr_tail()}"
            )
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as e:
            raise AssertionError(f"Server closed its input\n{self.stderr_tail()}") from e

    # ==================== Diagnostics ====================

    def stderr_tail(self, lines: int = 25) -> str:
        return "\n".join(self._stderr[-lines:]) or "(no stderr output)"

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)

    def __enter__(self) -> McpStdioClient:
        self.initialize()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ==================== Helpers ====================


def server_command(src_dir: Path, module: str = "session_search") -> list[str]:
    """The command that runs the MCP server as a module."""
    return [sys.executable, "-m", module]


def server_env(src_dir: Path, **overrides: str) -> dict[str, str]:
    """A full environment pointing the server at `src_dir` with overrides."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    env.update(overrides)
    return env
