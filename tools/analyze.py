#!/usr/bin/env python3
"""Download clustering artifacts from Comet ML and run analyzer(s).

Usage:
    # Single artifact
    python tools/analyze.py --artifact clustering-coach-CollegeMsg

    # Analyze all clustering artifacts (can be filtered by mode)
    python tools/analyze.py --all-artifacts --benchmark-mode dynamic

    # Config-driven run (see config/analyzer.yaml)
    python tools/analyze.py --config config/analyzer.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Allow `python tools/analyze.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer.artifacts import list_artifacts
from src.analyzer.config import AnalyzerConfigManager
from src.analyzer.runner import ANALYZERS, run


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _artifact_name_from_item(item: Dict[str, Any]) -> str | None:
    for key in ("name", "artifactName", "artifact_name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            # Some APIs may return workspace/name:version
            name = value.split("/")[-1]
            if ":" in name:
                name = name.split(":", 1)[0]
            return name
    return None


def _artifact_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _matches_benchmark_mode(item: Dict[str, Any], benchmark_mode: str | None) -> bool:
    if benchmark_mode is None:
        return True
    metadata = _artifact_metadata(item)
    value = metadata.get("benchmark_mode") or item.get("benchmark_mode")
    if not isinstance(value, str):
        # Unknown mode in listing response: keep it and let runtime output decide.
        return True
    return value == benchmark_mode


def _default_report_path(report_dir: Path, artifact_name: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{artifact_name}.json"


def _resolve_artifacts(
    args: argparse.Namespace,
    cfg: AnalyzerConfigManager,
    workspace: str,
) -> List[str]:
    explicit_from_cli = args.artifact or []
    if explicit_from_cli:
        return explicit_from_cli

    if args.artifact_group:
        group = cfg.artifact_group(args.artifact_group)
        if not group:
            raise ValueError(f"Artifact group not found or empty: {args.artifact_group}")
        return group

    selection_cfg = cfg.selection()
    default_mode = selection_cfg.get("default_mode", "explicit")
    default_group = selection_cfg.get("default_group", "")

    should_list_all = args.all_artifacts or default_mode == "all"

    if should_list_all:
        artifact_type = selection_cfg.get("artifact_type", "clustering-result")
        listed = list_artifacts(
            workspace=workspace,
            artifact_type=str(artifact_type),
        )
        mode_filter = args.benchmark_mode
        if mode_filter is None:
            filters_cfg = selection_cfg.get("filters", {})
            if not isinstance(filters_cfg, dict):
                filters_cfg = {}
            cfg_modes = _as_str_list(filters_cfg.get("benchmark_mode", []))
            if len(cfg_modes) == 1:
                mode_filter = cfg_modes[0]

        names = []
        for item in listed:
            if not isinstance(item, dict):
                continue
            if not _matches_benchmark_mode(item, mode_filter):
                continue
            name = _artifact_name_from_item(item)
            if name:
                names.append(name)
        return sorted(set(names))

    if default_mode == "group" and isinstance(default_group, str) and default_group:
        group = cfg.artifact_group(default_group)
        if group:
            return group

    explicit_cfg = cfg.explicit_artifacts()
    return explicit_cfg


def main():
    analyzer_choices = sorted(ANALYZERS.keys())

    parser = argparse.ArgumentParser(
        description="Download clustering artifact(s) and run analysis",
    )
    parser.add_argument(
        "--config",
        default="config/analyzer.yaml",
        help="Analyzer YAML config path (default: config/analyzer.yaml)",
    )
    parser.add_argument(
        "--workspace",
        required=False,
        help="Comet ML workspace name (overrides config)",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact name (repeatable, e.g. clustering-coach-CollegeMsg)",
    )
    parser.add_argument(
        "--artifact-group",
        default=None,
        help="Artifact group name from config selection.artifact_groups",
    )
    parser.add_argument(
        "--all-artifacts",
        action="store_true",
        help="Analyze all artifacts listed in the workspace",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["dynamic", "static"],
        default=None,
        help="Filter listed artifacts by benchmark mode metadata",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to download artifact(s) into (default from config)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Artifact version or alias (default: config or latest)",
    )
    parser.add_argument(
        "--analyzer",
        default=None,
        choices=analyzer_choices,
        help=f"Analyzer to run (default from config). Choices: {analyzer_choices}",
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
        help="Save analysis result. For multiple artifacts, provide a directory path",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    cfg = AnalyzerConfigManager(args.config)
    cfg_dirs = cfg.directories()
    cfg_analysis = cfg.analysis()

    workspace = args.workspace or cfg.workspace() or os.getenv("COMET_WORKSPACE")
    if not workspace:
        print(
            "Error: workspace not provided. Set --workspace, config workspace, or COMET_WORKSPACE.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build analyzer kwargs from CLI flags
    analyze_kwargs = {}
    configured_k = (
        cfg_analysis.get("overlap_quality", {}).get("betweenness_k")
        if isinstance(cfg_analysis.get("overlap_quality", {}), dict)
        else None
    )
    betweenness_k = args.betweenness_k if args.betweenness_k is not None else configured_k
    if betweenness_k is not None:
        analyze_kwargs["betweenness_k"] = int(betweenness_k)

    analyzer_name = args.analyzer or str(cfg_analysis.get("default_analyzer", "summary"))
    version_or_alias = args.version or cfg_analysis.get("default_version")

    output_dir: Path | None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cfg_dirs["artifact_download_dir"]

    try:
        artifacts = _resolve_artifacts(args, cfg, workspace)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Preserve order while removing duplicates
    artifacts = list(dict.fromkeys(artifacts))

    if not artifacts:
        print(
            "Error: no artifacts selected. Use --artifact, --artifact-group, --all-artifacts, or config selection.",
            file=sys.stderr,
        )
        sys.exit(1)

    save_json_enabled = bool(cfg_analysis.get("save_json", False))

    # Determine save target semantics
    save_path_arg = Path(args.save_json) if args.save_json else None
    multiple = len(artifacts) > 1
    if multiple and save_path_arg is not None and save_path_arg.suffix == ".json":
        print(
            "Error: --save-json must be a directory when analyzing multiple artifacts.",
            file=sys.stderr,
        )
        sys.exit(1)

    successes = 0
    failures = 0
    try:
        for artifact_name in artifacts:
            print(f"\n=== [{artifact_name}] ===")
            try:
                result = run(
                    workspace=workspace,
                    artifact_name=artifact_name,
                    output_dir=output_dir,
                    version_or_alias=(
                        str(version_or_alias) if version_or_alias is not None else None
                    ),
                    analyzer_name=analyzer_name,
                    **analyze_kwargs,
                )
            except Exception as exc:
                failures += 1
                print(f"Error analyzing {artifact_name}: {exc}", file=sys.stderr)
                continue

            report_out: Path | None = None
            if save_path_arg is not None:
                if save_path_arg.suffix == ".json":
                    report_out = save_path_arg
                else:
                    report_out = _default_report_path(save_path_arg, artifact_name)
            elif save_json_enabled:
                report_out = _default_report_path(cfg_dirs["report_dir"], artifact_name)

            if report_out is not None:
                report_out.parent.mkdir(parents=True, exist_ok=True)
                with open(report_out, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=2, default=str)
                print(f"Report saved to: {report_out}")

            successes += 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCompleted: {successes} succeeded, {failures} failed")
    if successes == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
