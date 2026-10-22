"""The gateway: the only service that is exposed. vLLM sits behind it.

`POST /v1/{platform}/process` takes a document and returns the booking items and lines, or the
reasons it needs review. The API key says which platforms a caller may use. A platform is served
by the adapter of the same name in `configs/serve.yaml`, and a request cannot name another
adapter or model. Every failure of the model ends in a review, never in an automated booking.
"""

import re
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from adapt.client import ModelClient
from adapt.config import Platform, load_platform
from adapt.gateway.keys import ApiKey, KeyFile, find_key
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
    *,
    platforms: Mapping[str, Platform] | None = None,
    timeout: float = 120.0,
) -> FastAPI:
    client = ModelClient(vllm_url, timeout=timeout)
    pipelines = build_pipelines(serve, client, platforms)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        client.close()

    app = FastAPI(title="Booking gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.serve = serve
    app.state.keys = keys
    app.state.pipelines = pipelines

    def require_key(authorization: Annotated[str | None, Header()] = None) -> ApiKey:
        scheme, _, presented = (authorization or "").partition(" ")
        key = find_key(keys, presented.strip()) if scheme.lower() == "bearer" else None
        if key is None:
            raise HTTPException(
                status_code=401,
                detail="missing or unknown API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return key

    @app.post("/v1/{platform}/process", response_model=ProcessResponse)
    def process(
        platform: str,
        body: ProcessRequest,
        key: Annotated[ApiKey, Depends(require_key)],
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> ProcessResponse:
        # A key without the platform gets the same answer whether the platform exists or not.
        pipeline = pipelines.get(platform)
        if platform not in key.platforms or pipeline is None:
            raise HTTPException(status_code=403, detail="this key may not use this platform")
        request_id = x_request_id if x_request_id and REQUEST_ID.match(x_request_id) else None
        result = pipeline.process(body.document_text, body.chart, request_id or uuid.uuid4().hex)
        return response_from(result)

    return app
