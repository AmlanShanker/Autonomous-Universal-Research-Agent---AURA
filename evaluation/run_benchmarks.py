from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.benchmark_runner import DEFAULT_BENCHMARK_DIR, run_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AURA's offline research evaluation benchmarks.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing versioned benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the complete machine-readable result JSON.",
    )
    args = parser.parse_args()

    result = run_benchmarks(args.benchmark_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
