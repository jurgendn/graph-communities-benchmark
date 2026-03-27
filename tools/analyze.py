#!/usr/bin/env python3
"""
Download a clustering artifact from Comet ML and run analysis.

Usage:
    # Basic summary (default)
    python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg

    # Overlap quality analysis
    python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \\
        --analyzer overlap-quality

    # With approximate betweenness centrality and JSON export
    python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \\
        --analyzer overlap-quality --betweenness-k 500 --save-json report.json
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow `python tools/analyze.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer.runner import ANALYZERS, run


def main():
    analyzer_choices = sorted(ANALYZERS.keys())

    parser = argparse.ArgumentParser(
        description="Download clustering artifact and run analysis",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Comet ML workspace name",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Artifact name (e.g. clustering-coach-CollegeMsg)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to download artifact into (default: temp dir)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Artifact version or alias (default: latest)",
    )
    parser.add_argument(
        "--analyzer",
        default="summary",
        choices=analyzer_choices,
        help=f"Analyzer to run (default: summary). Choices: {analyzer_choices}",
    )
    parser.add_argument(
        "--betweenness-k",
        type=int,
        default=None,
        help="Sample size for approximate betweenness centrality "
        "(default: None = exact). Only used by overlap-quality analyzer.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Save analysis result as JSON to this path",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    # Build analyzer kwargs from CLI flags
    analyze_kwargs = {}
    if args.betweenness_k is not None:
        analyze_kwargs["betweenness_k"] = args.betweenness_k

    try:
        result = run(
            workspace=args.workspace,
            artifact_name=args.artifact,
            output_dir=args.output_dir,
            version_or_alias=args.version,
            analyzer_name=args.analyzer,
            **analyze_kwargs,
        )

        if args.save_json:
            out_path = Path(args.save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=str)
            print(f"\nReport saved to: {out_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
