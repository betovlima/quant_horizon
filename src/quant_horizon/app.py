"""FastAPI application and browser CORS configuration."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import API_VERSION
from .exceptions import BusinessRuleError


app = FastAPI(
    title="Quant Horizon API",
    description=(
        "Generates quantitative signals and backtests. The API does not send "
        "brokerage orders and does not provide financial advice."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(BusinessRuleError)
async def business_rule_exception_handler(
    request: Request,
    exception: BusinessRuleError,
) -> JSONResponse:
    """Return stable, structured responses for business-rule violations."""

    del request
    return JSONResponse(
        status_code=exception.status_code,
        content={
            "error": {
                "code": exception.code,
                "message": exception.message,
                "context": exception.context,
            }
        },
    )


# The React interface runs on a different origin from the local API. The API
# starts on 127.0.0.1 by default and does not use cookies, so only the methods
# and headers required by the browser client are enabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


__all__ = ["app"]
