"""Unit tests for src/transport/security.py.

Tests SecurityConfig defaults/custom and LocalhostOnlyMiddleware
connection filtering behaviour.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import Response

from transport.security import LocalhostOnlyMiddleware, SecurityConfig


# ---------------------------------------------------------------------------
# SecurityConfig
# ---------------------------------------------------------------------------


class TestSecurityConfig:
    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.localhost_only is True
        assert cfg.require_api_key is False
        assert cfg.api_key is None

    def test_default_allowed_origins_populated(self):
        cfg = SecurityConfig()
        assert len(cfg.allowed_origins) > 0
        assert "http://localhost" in cfg.allowed_origins
        assert "http://127.0.0.1" in cfg.allowed_origins

    def test_custom_allowed_origins_not_overwritten(self):
        cfg = SecurityConfig(allowed_origins=["http://myapp.test"])
        # __post_init__ only fills in defaults when list is empty
        assert cfg.allowed_origins == ["http://myapp.test"]

    def test_custom_values_preserved(self):
        cfg = SecurityConfig(localhost_only=False, require_api_key=True, api_key="secret")
        assert cfg.localhost_only is False
        assert cfg.require_api_key is True
        assert cfg.api_key == "secret"


# ---------------------------------------------------------------------------
# LocalhostOnlyMiddleware
# ---------------------------------------------------------------------------


def _make_request(host: str | None) -> MagicMock:
    """Build a mock Starlette Request with a given client host."""
    request = MagicMock()
    if host is None:
        request.client = None
    else:
        request.client = MagicMock()
        request.client.host = host
    return request


async def _dummy_call_next(request):
    return Response(content="OK", status_code=200)


class TestLocalhostOnlyMiddleware:
    def setup_method(self):
        # BaseHTTPMiddleware requires an `app` argument
        app = MagicMock()
        self.middleware = LocalhostOnlyMiddleware(app=app)

    async def test_localhost_ipv4_allowed(self):
        request = _make_request("127.0.0.1")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 200

    async def test_localhost_ipv6_allowed(self):
        request = _make_request("::1")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 200

    async def test_localhost_name_allowed(self):
        request = _make_request("localhost")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 200

    async def test_remote_ip_rejected(self):
        request = _make_request("192.168.1.42")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 403

    async def test_public_ip_rejected(self):
        request = _make_request("8.8.8.8")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 403

    async def test_none_client_rejected(self):
        request = _make_request(None)
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 403

    async def test_rejected_response_is_plain_text(self):
        request = _make_request("10.0.0.1")
        response = await self.middleware.dispatch(request, _dummy_call_next)
        assert response.status_code == 403
        assert response.media_type == "text/plain"

    async def test_allowed_hosts_set_contains_expected_values(self):
        assert "127.0.0.1" in LocalhostOnlyMiddleware.ALLOWED_HOSTS
        assert "::1" in LocalhostOnlyMiddleware.ALLOWED_HOSTS
        assert "localhost" in LocalhostOnlyMiddleware.ALLOWED_HOSTS
