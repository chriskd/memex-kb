"""No-op Chroma telemetry implementation."""

from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopProductTelemetry(ProductTelemetryClient):
    """Drop Chroma product telemetry events without importing PostHog."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
