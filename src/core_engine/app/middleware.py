"""Bound multipart bodies before the framework spools them to disk."""
from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from core_engine.report.documents import MAX_FILE_BYTES


class UploadLimitMiddleware:
    def __init__(self, app, max_bytes: int = MAX_FILE_BYTES + 64 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        template_upload = path == "/api/templates"
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or not (template_upload or path.startswith("/api/database/") and path.endswith("/articles"))):
            return await self.app(scope, receive, send)
        limit = 10 * 1024 * 1024 + 64 * 1024 if template_upload else self.max_bytes
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.max_bytes + 1
        if length < 0 or length > limit:
            response = JSONResponse({"detail": "Upload body exceeds the size limit."}, status_code=413)
            return await response(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail="Upload body exceeds the size limit.")
            return message

        await self.app(scope, limited_receive, send)
