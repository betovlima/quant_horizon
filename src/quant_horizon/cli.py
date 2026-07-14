"""Command-line entry point for the Quant Horizon API."""

from __future__ import annotations

import argparse
import os

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    """Create the API command-line parser."""
    parser = argparse.ArgumentParser(description="Run the Quant Horizon API.")
    parser.add_argument(
        "--host",
        default=os.environ.get("QUANT_HORIZON_HOST", "127.0.0.1"),
        help="Host interface used by Uvicorn.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.environ.get("PORT", os.environ.get("QUANT_HORIZON_PORT", "8000"))
        ),
        help="TCP port used by Uvicorn.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Environment file loaded by Uvicorn.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the server when Python files change.",
    )
    return parser


def main() -> None:
    """Run the FastAPI application through Uvicorn."""
    arguments = build_parser().parse_args()
    uvicorn.run(
        "quant_horizon.api:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        env_file=arguments.env_file,
    )


if __name__ == "__main__":
    main()
