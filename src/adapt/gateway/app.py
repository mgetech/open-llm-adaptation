"""The gateway: the only service that is exposed. vLLM sits behind it.

`POST /v1/{platform}/process` takes a document and returns the booking items and lines, or the
reasons it needs review. The API key says which platforms a caller may use. A platform is served
by the adapter of the same name in `configs/serve.yaml`, and a request cannot name another
adapter or model. Every failure of the model ends in a review, never in an automated booking.
"""

import hashlib
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from adapt.client import ModelClient
from adapt.config import Platform, load_platform
from adapt.data.pseudonymise import pseudonymise
from adapt.gateway.audit import AuditLog, AuditRecord, Outcome, now
from adapt.gateway.health import AdapterHashes, Metrics, readiness
from adapt.gateway.keys import ApiKey, KeyFile, find_key
from adapt.models import Registry, load_registry
from adapt.pipeline import Pipeline, Result
from adapt.serve import ServeConfig

MAX_BODY_CHARACTERS = 100_000
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
HEADER_FIELDS = (
    "supplier",
    "document_date",
    "document_type",
    "currency",
    "total",
    "reverse_charge",
)


class ProcessRequest(BaseModel):
    document_text: str = Field(max_length=MAX_BODY_CHARACTERS)
    chart: str = Field(max_length=64)


class BookingLineOut(BaseModel):
    account: str
    vat_rate: str
    gross: str
    net: str
    vat: str
    liability: str


class ProcessResponse(BaseModel):
    request_id: str
    needs_review: bool
    reasons: list[str]
    document: dict[str, Any] | None
    items: list[dict[str, Any]]
    booking_lines: list[BookingLineOut]
    adapter: str
    adapter_version: str | None
    confidence: float | None


def _money(value: Decimal) -> str:
    return f"{value:.2f}"


def response_from(result: Result) -> ProcessResponse:
    output = result.output
    return ProcessResponse(
        request_id=result.request_id,
        needs_review=result.needs_review,
        reasons=list(result.reasons),
        document={name: output[name] for name in HEADER_FIELDS if name in output}
        if output
        else None,
        items=result.items,
        booking_lines=[
            BookingLineOut(
                account=line.account,
                vat_rate=line.vat_rate,
                gross=_money(line.gross),
                net=_money(line.net),
                vat=_money(line.vat),
                liability=_money(line.liability),
            )
            for line in result.booking_lines
        ],
        adapter=result.adapter,
        adapter_version=result.adapter_version,
        confidence=result.confidence,
    )


def build_pipelines(
    serve: ServeConfig, client: ModelClient, platforms: Mapping[str, Platform] | None = None
) -> dict[str, Pipeline]:
    """One pipeline for each adapter, for the platform that has the adapter's name."""
    pipelines = {}
    for adapter in serve.adapters:
        platform = (platforms or {}).get(adapter.name) or load_platform(adapter.name)
        pipelines[adapter.name] = Pipeline(
            platform,
            client,
            adapter=adapter.name,
            threshold=adapter.threshold,
            adapter_version=adapter.sha256,
        )
    return pipelines


def create_app(
    serve: ServeConfig,
    keys: KeyFile,
    vllm_url: str,
    audit: AuditLog,
    *,
    platforms: Mapping[str, Platform] | None = None,
    registry: Registry | None = None,
    timeout: float = 120.0,
) -> FastAPI:
    client = ModelClient(vllm_url, timeout=timeout)
    pipelines = build_pipelines(serve, client, platforms)
    base_revision = (registry or load_registry()).models[serve.model].revision
    metrics = Metrics(list(pipelines))
    hashes = AdapterHashes()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        client.close()

    app = FastAPI(title="Booking gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.serve = serve
    app.state.keys = keys
    app.state.pipelines = pipelines
    app.state.audit = audit
    app.state.metrics = metrics

    def log(
        request_id: str, platform: str, outcome: Outcome, started: float, **fields: Any
    ) -> None:
        """Write the record of a request and count it. If the record cannot be written, the
        request fails."""
        seconds = time.monotonic() - started
        try:
            audit.write(
                AuditRecord(
                    timestamp=now(),
                    request_id=request_id,
                    platform=platform,
                    outcome=outcome,
                    latency_ms=round((time.monotonic() - started) * 1000),
                    **{
                        "key_id": None,
                        "input_sha256": None,
                        "adapter": None,
                        "adapter_sha256": None,
                        "base_revision": None,
                        "prompt_sha256": None,
                        "review_reasons": [],
                    }
                    | fields,
                )
            )
        except OSError as error:
            metrics.observe(platform, "error", [], seconds)
            raise HTTPException(
                status_code=500, detail="the audit log cannot be written"
            ) from error
        metrics.observe(platform, outcome, fields.get("review_reasons", []), seconds)

    def require_key(
        platform: str,
        authorization: Annotated[str | None, Header()] = None,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> ApiKey:
        started = time.monotonic()
        scheme, _, presented = (authorization or "").partition(" ")
        key = find_key(keys, presented.strip()) if scheme.lower() == "bearer" else None
        if key is None:
            log(request_id_of(x_request_id), platform, "unauthorized", started)
            raise HTTPException(
                status_code=401,
                detail="missing or unknown API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return key

    def request_id_of(header: str | None) -> str:
        return header if header and REQUEST_ID.match(header) else uuid.uuid4().hex

    @app.post("/v1/{platform}/process", response_model=ProcessResponse)
    def process(
        platform: str,
        body: ProcessRequest,
        key: Annotated[ApiKey, Depends(require_key)],
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> ProcessResponse:
        started = time.monotonic()
        request_id = request_id_of(x_request_id)
        # A key without the platform gets the same answer whether the platform exists or not.
        pipeline = pipelines.get(platform)
        if platform not in key.platforms or pipeline is None:
            log(request_id, platform, "forbidden", started, key_id=key.id)
            raise HTTPException(status_code=403, detail="this key may not use this platform")
        result = pipeline.process(body.document_text, body.chart, request_id)
        log(
            request_id,
            platform,
            "review" if result.needs_review else "automated",
            started,
            key_id=key.id,
            input_sha256=hashlib.sha256(pseudonymise(body.document_text).encode()).hexdigest(),
            adapter=result.adapter,
            adapter_sha256=result.adapter_version,
            base_revision=base_revision,
            prompt_sha256=pipeline.prompt_sha256,
            review_reasons=[str(reason) for reason in result.reasons],
        )
        return response_from(result)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        problems = readiness(serve, client, hashes)
        body = {"ready": not problems, "problems": problems}
        return JSONResponse(body, status_code=503 if problems else 200)

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
