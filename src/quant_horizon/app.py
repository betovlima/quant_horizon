"""FastAPI application and browser CORS configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_VERSION


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
