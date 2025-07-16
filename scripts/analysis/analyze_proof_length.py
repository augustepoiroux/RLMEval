#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import jsonlines
from lean_interact.utils import remove_lean_comments
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rlm_eval.utils import ROOT_DIR

console = Console()


def clean_proof(proof: str) -> str:
    """Clean the proof text by removing comments and extra spaces."""
    # Remove comments
    clean_proof = remove_lean_comments(proof)
    if not clean_proof:
        clean_proof = proof

    # Remove potential code following the proof
    proof_lines = clean_proof.split("\n")
    cleaned_lines = []
    for line in proof_lines:
        if not line.strip():
            continue
        if line.lstrip() == line:
            break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


@dataclass
class ValidProofStats:
    """Statistics about a valid proof."""

    length_chars: int  # Length in characters
    length_lines: int  # Length in lines
    num_tokens: int | None = None  # Number of tokens (if tokenizer available)


@dataclass
class FileStats:
    """Statistics for a single validity_results.jsonl file."""

    path: str
    model: str
    timestamp: str
    project: str
    avg_chars: float
    avg_lines: float
    median_chars: float
    median_lines: float
    min_lines: int
    max_lines: int
    std_chars: float
    std_lines: float
    count: int


def count_lines(text: str) -> int:
    """Count non-empty lines in a string."""
    return len([line for line in text.strip().split("\n") if line.strip()])


def analyze_validity_results(validity_path: str) -> list[ValidProofStats]:
    """Extract statistics about valid proofs from a validity_results.jsonl file."""
    result_stats = []

    try:
        with jsonlines.open(validity_path) as reader:
            for result in reader:
                # In proof evaluation, the field is 'lean_proof'
                lean_code = result.get("lean_proof")
                well_typed = result.get("well_typed", False)

                if lean_code and well_typed:
                    # Clean the proof text before calculating statistics
                    lean_code_no_comments = clean_proof(lean_code)
                    if not lean_code_no_comments:
                        lean_code_no_comments = lean_code

                    stats = ValidProofStats(
                        length_chars=len(lean_code_no_comments),
                        length_lines=count_lines(lean_code_no_comments),
                    )
                    result_stats.append(stats)
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing {validity_path}: {e}")

    return result_stats


def extract_path_info(path: str) -> tuple[str, str, str]:
    """Extract model, timestamp, and project name from path."""
    # Path structure: ROOT_DIR/results/proof/project_name/model_name/timestamp/...
    parts = path.split(os.path.sep)

    project = "unknown"
    model = "unknown"
    timestamp = "unknown"

    # Find the index of "proof" in the path
    if "proof" in parts:
        proof_idx = parts.index("proof")
        if proof_idx + 1 < len(parts):
            project = parts[proof_idx + 1]
        if proof_idx + 2 < len(parts):
            model = parts[proof_idx + 2]
        if proof_idx + 3 < len(parts):
            # Timestamp is typically in the format YYYYMMDD_HHMMSS
            timestamp_match = re.search(r"(\d{8}_\d{6}.*)", parts[proof_idx + 3])
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            else:
                timestamp = parts[proof_idx + 3]

    return model, timestamp, project


def find_validity_results(results_dir: str) -> list[str]:
    """Find all validity_results.jsonl files in the given directory."""
    return glob.glob(f"{results_dir}/**/validity_results.jsonl", recursive=True)


def calculate_file_statistics(path: str) -> FileStats | None:
    """Calculate statistics for a single validity_results.jsonl file."""
    stats_list = analyze_validity_results(path)

    if not stats_list:
        return None

    model, timestamp, project = extract_path_info(path)
    chars = [s.length_chars for s in stats_list]
    lines = [s.length_lines for s in stats_list]

    return FileStats(
        path=path,
        model=model,
        timestamp=timestamp,
        project=project,
        avg_chars=statistics.mean(chars),
        avg_lines=statistics.mean(lines),
        median_chars=statistics.median(chars),
        median_lines=statistics.median(lines),
        min_lines=min(lines),
        max_lines=max(lines),
        std_chars=statistics.stdev(chars) if len(chars) > 1 else 0,
        std_lines=statistics.stdev(lines) if len(lines) > 1 else 0,
        count=len(stats_list),
    )


def calculate_aggregate_statistics(file_stats: list[FileStats]) -> dict[str, Any]:
    """Calculate aggregate statistics from file-level stats."""
    if not file_stats:
        return {
            "count": 0,
            "file_count": 0,
            "chars": {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0},
            "lines": {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0},
        }

    # Calculate the overall mean as the weighted average of file means
    total_count = sum(fs.count for fs in file_stats)
    avg_chars = sum(fs.avg_chars * fs.count for fs in file_stats) / total_count if total_count > 0 else 0
    avg_lines = sum(fs.avg_lines * fs.count for fs in file_stats) / total_count if total_count > 0 else 0

    # For other metrics, collect values from all files
    all_median_chars = [fs.median_chars for fs in file_stats]
    all_median_lines = [fs.median_lines for fs in file_stats]
    all_min_lines = [fs.min_lines for fs in file_stats]
    all_max_lines = [fs.max_lines for fs in file_stats]

    return {
        "count": total_count,
        "file_count": len(file_stats),
        "chars": {
            "mean": avg_chars,
            "median": statistics.median(all_median_chars) if all_median_chars else 0,
            "min": min([fs.avg_chars for fs in file_stats]) if file_stats else 0,
            "max": max([fs.avg_chars for fs in file_stats]) if file_stats else 0,
            "std": statistics.stdev([fs.avg_chars for fs in file_stats]) if len(file_stats) > 1 else 0,
        },
        "lines": {
            "mean": avg_lines,
            "median": statistics.median(all_median_lines) if all_median_lines else 0,
            "min": min(all_min_lines) if all_min_lines else 0,
            "max": max(all_max_lines) if all_max_lines else 0,
            "std": statistics.stdev([fs.avg_lines for fs in file_stats]) if len(file_stats) > 1 else 0,
        },
    }


def process_validity_files(
    validity_paths: list[str], only_rechecked: bool = False
) -> dict[str, dict[str, list[FileStats]]]:
    """Process all validity files and group by model and timestamp."""
    # Structure: model -> timestamp -> [file_stats]
    model_timestamp_stats: dict[str, dict[str, list[FileStats]]] = defaultdict(lambda: defaultdict(list))

    for path in tqdm(validity_paths, desc="Processing files"):
        file_stats = calculate_file_statistics(path)
        if file_stats:
            # Skip if we're only looking for rechecked files and this one doesn't have _recheck suffix
            if only_rechecked and "_recheck" not in file_stats.timestamp:
                continue

            model_timestamp_stats[file_stats.model][file_stats.timestamp].append(file_stats)

    return model_timestamp_stats


def print_model_statistics(model_name: str, timestamp_stats: dict[str, list[FileStats]]) -> dict[str, Any]:
    """Print statistics for a single model grouped by timestamp."""
    table = Table(title=f"Proof Length Statistics for {model_name}")

    # Add columns
    table.add_column("Timestamp", style="yellow")
    table.add_column("Valid Proofs", style="green")
    table.add_column("Files", style="green")
    table.add_column("Mean Chars", style="magenta")
    table.add_column("Mean Lines", style="blue")
    table.add_column("Median Lines", style="blue")
    table.add_column("Min Lines", style="blue")
    table.add_column("Max Lines", style="blue")

    # Sort timestamps in chronological order
    sorted_timestamps = sorted(timestamp_stats.keys())

    model_results = {}
    all_file_stats = []

    # Add a row for each timestamp
    for timestamp in sorted_timestamps:
        file_stats = timestamp_stats[timestamp]
        all_file_stats.extend(file_stats)

        stats = calculate_aggregate_statistics(file_stats)
        model_results[timestamp] = stats

        table.add_row(
            timestamp,
            str(stats["count"]),
            str(stats["file_count"]),
            f"{stats['chars']['mean']:.1f}",
            f"{stats['lines']['mean']:.1f}",
            f"{stats['lines']['median']:.1f}",
            str(stats["lines"]["min"]),
            str(stats["lines"]["max"]),
        )

    # Add overall row for this model
    if len(sorted_timestamps) > 1:
        overall_stats = calculate_aggregate_statistics(all_file_stats)
        model_results["overall"] = overall_stats
        table.add_section()
        table.add_row(
            "OVERALL",
            str(overall_stats["count"]),
            str(overall_stats["file_count"]),
            f"{overall_stats['chars']['mean']:.1f}",
            f"{overall_stats['lines']['mean']:.1f}",
            f"{overall_stats['lines']['median']:.1f}",
            str(overall_stats["lines"]["min"]),
            str(overall_stats["lines"]["max"]),
        )

    console.print(table)
    return model_results


def print_overall_statistics(model_timestamp_stats: dict[str, dict[str, list[FileStats]]]) -> dict[str, Any]:
    """Print overall statistics across all models."""
    # Flatten all file stats
    all_file_stats = []
    for model_name, timestamp_stats in model_timestamp_stats.items():
        for timestamp, file_stats in timestamp_stats.items():
            all_file_stats.extend(file_stats)

    # Calculate overall stats
    overall_stats = calculate_aggregate_statistics(all_file_stats)

    # Create a summary table
    table = Table(title="Overall Proof Length Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Models", str(len(model_timestamp_stats)))
    table.add_row("Total Timestamps", str(sum(len(ts) for ts in model_timestamp_stats.values())))
    table.add_row("Total Files", str(overall_stats["file_count"]))
    table.add_row("Total Valid Proofs", str(overall_stats["count"]))
    table.add_row("Average Characters", f"{overall_stats['chars']['mean']:.1f}")
    table.add_row("Average Lines", f"{overall_stats['lines']['mean']:.1f}")
    table.add_row("Median Lines", f"{overall_stats['lines']['median']:.1f}")
    table.add_row("Min Lines", str(overall_stats["lines"]["min"]))
    table.add_row("Max Lines", str(overall_stats["lines"]["max"]))

    console.print(table)
    return overall_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze lengths of valid proofs from evaluation runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "proof"),
        help="Directory containing evaluation results",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save the results JSON (optional)")
    parser.add_argument("--model", type=str, default=None, help="Only show results for a specific model")
    parser.add_argument(
        "--only-rechecked", action="store_true", help="Only include results from timestamps with '_recheck' suffix"
    )
    args = parser.parse_args()

    console.print(f"[bold]Looking for validity results in: {args.results_dir}")

    # Find all validity_results.jsonl files
    validity_paths = find_validity_results(args.results_dir)
    console.print(f"[green]Found {len(validity_paths)} validity result files")

    if not validity_paths:
        console.print("[yellow]No validity result files found. Did you specify the correct directory?")
        return

    # Process files and group by model and timestamp
    model_timestamp_stats = process_validity_files(validity_paths, args.only_rechecked)

    # Results to save
    results = {"models": {}, "overall": {}}

    # If specific model requested, only show that model
    if args.model and args.model in model_timestamp_stats:
        results["models"][args.model] = print_model_statistics(args.model, model_timestamp_stats[args.model])
    else:
        # Print statistics for each model
        for model_name, timestamp_stats in model_timestamp_stats.items():
            results["models"][model_name] = print_model_statistics(model_name, timestamp_stats)
            console.print()  # Add space between tables

    # Print overall statistics
    results["overall"] = print_overall_statistics(model_timestamp_stats)

    # Save results if output path specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to: {args.output}")


if __name__ == "__main__":
    main()
