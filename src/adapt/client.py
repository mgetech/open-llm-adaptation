"""Client for an OpenAI-compatible chat endpoint (vLLM) with JSON-schema output and logprobs."""

import json
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import TracebackType
from typing import Any, Literal, Self

import httpx
from jsonschema import Draft202012Validator

TIMEOUT_SECONDS = 30.0
RETRIES = 1


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    content: str
    output: dict[str, Any]
    tokens: list[TokenLogprob]


class ModelError(Exception):
    """The model did not give a usable answer, even after the retry.

    `kind` is "unavailable" for a timeout, a connection problem or an HTTP error, and
    "invalid_output" for an answer that is not JSON or does not fit the schema.
    """

    def __init__(self, kind: Literal["unavailable", "invalid_output"], message: str) -> None:
        super().__init__(message)
        self.kind = kind


@cache
def ssl_context() -> ssl.SSLContext:
    """Loading the certificate bundle takes about half a second, so it is done once."""
    return httpx.create_ssl_context()


class ModelClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        retries: int = RETRIES,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", headers=headers, verify=ssl_context()
        )
        if retries < 0:
            raise ValueError("retries must not be negative")
        self._timeout = timeout
        self._retries = retries

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def models(self, timeout: float = 5.0) -> list[str]:
        """The names the server serves: the base model and its adapters."""
        try:
            response = self._http.get("models", timeout=timeout)
        except httpx.HTTPError as error:
            raise ModelError("unavailable", f"server not reachable: {error!r}") from error
        if response.status_code != 200:
            raise ModelError("unavailable", f"the model server returned {response.status_code}")
        try:
            return [entry["id"] for entry in response.json()["data"]]
        except (ValueError, KeyError, TypeError) as error:
            raise ModelError("invalid_output", f"unexpected response shape: {error!r}") from error

    def complete(
        self, model: str, messages: Sequence[Mapping[str, str]], schema: Mapping[str, Any]
    ) -> Completion:
        """Ask `model` (the adapter name) for one answer that fits `schema`.

        A failed attempt is repeated up to `retries` times. If the last attempt also fails,
        the error of that attempt is raised.
        """
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": 0,
            "logprobs": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "booking", "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        validator = Draft202012Validator(schema)
        for attempt in range(self._retries + 1):
            try:
                return self._attempt(payload, validator)
            except ModelError as error:
                if attempt == self._retries:
                    raise error
        raise AssertionError("unreachable")

    def _attempt(self, payload: dict[str, Any], validator: Draft202012Validator) -> Completion:
        try:
            response = self._http.post("chat/completions", json=payload, timeout=self._timeout)
        except httpx.TimeoutException as error:
            raise ModelError("unavailable", f"no answer in time: {error!r}") from error
        except httpx.HTTPError as error:
            raise ModelError("unavailable", f"server not reachable: {error!r}") from error
        if response.status_code != 200:
            raise ModelError("unavailable", f"the model server returned {response.status_code}")
        return _parse(response, validator)


def _parse(response: httpx.Response, validator: Draft202012Validator) -> Completion:
    try:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ModelError("invalid_output", f"unexpected response shape: {error!r}") from error
    if not isinstance(content, str):
        raise ModelError("invalid_output", "the answer has no text")
    if choice.get("finish_reason") == "length":
        raise ModelError("invalid_output", "the answer was cut off")
    try:
        output = json.loads(content)
    except ValueError as error:
        raise ModelError("invalid_output", f"the answer is not valid JSON: {error}") from error
    if not isinstance(output, dict):
        raise ModelError("invalid_output", "the answer is not a JSON object")
    problem = next(iter(validator.iter_errors(output)), None)
    if problem is not None:
        raise ModelError("invalid_output", f"the answer does not fit the schema: {problem.message}")
    try:
        tokens = _tokens(choice)
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError("invalid_output", f"the logprobs are malformed: {error!r}") from error
    return Completion(content=content, output=output, tokens=tokens)


def _tokens(choice: Mapping[str, Any]) -> list[TokenLogprob]:
    """The generated tokens with their log-probabilities, or an empty list if none came back."""
    entries = (choice.get("logprobs") or {}).get("content") or []
    return [TokenLogprob(entry["token"], float(entry["logprob"])) for entry in entries]
