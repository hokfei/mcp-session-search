"""Verify the server starts with the exact command and env from mcp.json.

Reads ~/.opencode/mcp.json, spawns the session-search server with those exact
values, and confirms it (a) starts, (b) sees the configured paths, and
(c) answers a real search. This is the wiring test: if it passes, OpenCode will
be able to reach this server.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

MCP_JSON = Path.home() / ".opencode" / "mcp.json"
SERVER_KEY = "session-search"

EXPECTED = {
    "model_path": "bge-small-en-v1.5-q8_0.gguf",
    "opencode_db_path": "opencode.db",
    "database_path": "session_search.db",
    "embedding_dim": 384,
}


def read_server_config() -> dict:
    config = json.loads(MCP_JSON.read_text(encoding="utf-8"))
    try:
        return config["mcpServers"][SERVER_KEY]
    except KeyError as e:
        raise SystemExit(
            f"No {SERVER_KEY!r} entry in {MCP_JSON}. Found: "
            f"{sorted(config['mcpServers'])}"
        ) from e


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    spec = read_server_config()
    print(f"mcp.json entry found: {SERVER_KEY}")
    print(f"  command: {spec['command']}")
    print(f"  args:    {spec['args']}")

    command = Path(spec["command"])
    if not command.exists():
        fail(f"interpreter missing: {command}")

    env = dict(os.environ)
    env.update(spec.get("env", {}))

    # The prefix fix is what makes these variables do anything at all.
    for key, value in spec.get("env", {}).items():
        if key.startswith("SESSION_SEARCH_"):
            setting = key[len("SESSION_SEARCH_") :].lower()
            print(f"  env: {setting} = {value}")

    proc = subprocess.Popen(
        [str(command), *spec["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    stderr: list[str] = []
    threading.Thread(
        target=lambda: [stderr.append(line.rstrip()) for line in proc.stderr],
        daemon=True,
    ).start()

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def read_for(request_id: int, timeout: int = 180) -> dict:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                fail(f"server closed stdout\n" + "\n".join(stderr[-20:]))
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == request_id:
                return payload
        fail(f"timed out waiting for response {request_id}\n" + "\n".join(stderr[-20:]))

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "verify", "version": "1.0"},
                },
            }
        )
        init = read_for(1)
        server_info = init["result"]["serverInfo"]
        print(f"\nserver: {server_info['name']}")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = read_for(2)["result"]["tools"]
        print(f"tools: {', '.join(t['name'] for t in tools)}")

        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "health_check", "arguments": {}},
            }
        )
        health = json.loads(read_for(3)["result"]["content"][0]["text"])
        print(f"\nhealth: {json.dumps(health, indent=2)}")

        if health["status"] != "healthy":
            fail(f"server reported {health['status']}")
        if health["embedding_model"] != spec["env"]["SESSION_SEARCH_MODEL_PATH"]:
            fail(
                "embedding model from mcp.json was ignored:\n"
                f"  configured {spec['env']['SESSION_SEARCH_MODEL_PATH']}\n"
                f"  actual     {health['embedding_model']}"
            )
        if health["opencode_db"] != spec["env"]["SESSION_SEARCH_OPENCODE_DB_PATH"]:
            fail(
                "opencode db from mcp.json was ignored:\n"
                f"  configured {spec['env']['SESSION_SEARCH_OPENCODE_DB_PATH']}\n"
                f"  actual     {health['opencode_db']}"
            )

        print("\nendpoints verified: model path and opencode db both honoured")

        send(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "search_sessions",
                    "arguments": {"query": "which python version is installed", "top_k": 3},
                },
            }
        )
        search = json.loads(read_for(4, timeout=300)["result"]["content"][0]["text"])
        print(f"\nsearch returned {search['match_count']} match(es):")
        for match in search["matches"]:
            print(f"  {match['relevance']:.3f}  {match['title']}")

        if search["match_count"] == 0:
            print("\nWARNING: a relevant query returned nothing")

        print("\nOK")
        return 0

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())