"""AURA command-line entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aura",
        description="Run AURA services.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api", help="Start the FastAPI server.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")

    subparsers.add_parser("worker", help="Start the durable research worker.")

    args = parser.parse_args()
    if args.command == "api":
        import uvicorn

        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "worker":
        from workers.research_worker import main as worker_main

        worker_main()


__all__ = ["main"]
