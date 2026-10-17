"""A small OpenAI-compatible server for tests. It replaces vLLM and needs no GPU or network.

Queue replies with `server.reply(...)`. Each request takes the next reply in the queue; when
the queue is empty the server answers 500. Requests are kept in `server.requests`.

For tests that send many requests at once, set `server.handler` instead. It is called with the
request body and returns the `Reply`, so the answer does not depend on the order of requests.
Set `server.version` to make `GET /version` answer like vLLM does.
"""

import json
import re
import socketserver
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

DEFAULT_LOGPROB = -0.001
TOKEN_PATTERN = re.compile(r"\d+|[A-Za-z]+|\s+|.", re.DOTALL)


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    delay: float = 0.0
    drop: bool = False  # close the connection without an answer


@dataclass(frozen=True)
class Request:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


def split_tokens(text: str) -> list[str]:
    """Cut text into pieces like a tokenizer would: digit runs, words, spaces, single marks."""
    return TOKEN_PATTERN.findall(text)


def completion_body(
    content: str,
    logprob_for: Callable[[str], float] = lambda token: DEFAULT_LOGPROB,
    *,
    finish_reason: str = "stop",
    with_logprobs: bool = True,
) -> dict[str, Any]:
    """A chat completion with `content` and one logprob entry per token piece."""
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish_reason,
        "logprobs": None,
    }
    if with_logprobs:
        pieces = split_tokens(content)
        choice["logprobs"] = {
            "content": [
                {
                    "token": piece,
                    "logprob": logprob_for(piece),
                    "bytes": list(piece.encode()),
                    "top_logprobs": [],
                }
                for piece in pieces
            ]
        }
    return {"id": "chatcmpl-test", "object": "chat.completion", "choices": [choice]}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:
        # HTTPServer looks up the host name here, which is slow on some machines.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class FakeServer:
    def __init__(self, port: int = 0) -> None:
        self.requests: list[Request] = []
        self.handler: Callable[[dict[str, Any]], Reply] | None = None
        self.version: str | None = None
        self._replies: list[Reply] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with outer._lock:
                    outer.requests.append(
                        Request(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                    )
                    if outer.handler:
                        reply = outer.handler(body)
                    else:
                        reply = outer._replies.pop(0) if outer._replies else Reply(status=500)
                time.sleep(reply.delay)
                if reply.drop:
                    self.close_connection = True
                    return
                data = (
                    reply.body.encode()
                    if isinstance(reply.body, str)
                    else json.dumps(reply.body).encode()
                )
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up waiting

            def do_GET(self) -> None:
                found = self.path == "/version" and outer.version is not None
                data = json.dumps({"version": outer.version} if found else {}).encode()
                self.send_response(200 if found else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = _Server(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def reply(self, body: Any = None, *, status: int = 200, delay: float = 0.0) -> Self:
        with self._lock:
            self._replies.append(Reply(status, body, delay))
        return self

    def drop(self) -> Self:
        """Queue a reply that closes the connection without answering."""
        with self._lock:
            self._replies.append(Reply(drop=True))
        return self

    def reset(self) -> None:
        """Forget queued replies and recorded requests."""
        with self._lock:
            self._replies.clear()
            self.requests.clear()

    def wait_for_requests(self, count: int, timeout: float = 5.0) -> None:
        """Wait until `count` requests have arrived. A client that timed out may be ahead."""
        deadline = time.monotonic() + timeout
        while len(self.requests) < count and time.monotonic() < deadline:
            time.sleep(0.01)

    def reply_with_output(
        self, output: dict[str, Any], *, delay: float = 0.0, **kwargs: Any
    ) -> Self:
        """Queue a normal answer whose content is `output` written as JSON."""
        return self.reply(completion_body(json.dumps(output), **kwargs), delay=delay)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
