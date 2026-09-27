from fastapi import Request
from fastapi.responses import JSONResponse


class UpstreamError(Exception):
    """Erreur d'un micro-service en aval (502, ou 504 pour un timeout)."""

    def __init__(self, service: str, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.service = service
        self.detail = detail
        self.status_code = status_code


class InvalidUpload(Exception):
    """Fichier envoyé par le client invalide (422, ou 413 si trop gros)."""

    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def upstream_error_handler(_: Request, exc: UpstreamError) -> JSONResponse:
    error = "gateway_timeout" if exc.status_code == 504 else "bad_gateway"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error, "service": exc.service, "detail": exc.detail},
    )


async def invalid_upload_handler(_: Request, exc: InvalidUpload) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "invalid_upload", "detail": exc.detail},
    )
