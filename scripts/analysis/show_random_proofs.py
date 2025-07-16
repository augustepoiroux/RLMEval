#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

import jsonlines
from lean_interact.utils import remove_lean_comments
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from tqdm import tqdm

from rlm_eval.utils import ROOT_DIR

console = Console()


def find_validity_results(results_dir: str) -> List[str]:
    """Find all validity_results.jsonl files in the given directory."""
    return glob.glob(f"{results_dir}/**/validity_results.jsonl", recursive=True)


def extract_valid_proofs(validity_path: str) -> List[Dict[str, Any]]:
    """Extract valid proofs from a validity_results.jsonl file."""
    valid_proofs = []

    try:
        with jsonlines.open(validity_path) as reader:
            for result in reader:
                # In proof evaluation, the field is 'lean_proof'
                lean_code = result.get("lean_proof")
                well_typed = result.get("well_typed", False)

                if lean_code and well_typed:
                    # Extract path info to add to the result
                    path_parts = validity_path.split(os.path.sep)

                    # Extract project, model and timestamp if possible
                    project = "unknown"
                    model = "unknown"
                    timestamp = "unknown"

                    if "proof" in path_parts:
                        proof_idx = path_parts.index("proof")
                        if proof_idx + 1 < len(path_parts):
                            project = path_parts[proof_idx + 1]
                        if proof_idx + 2 < len(path_parts):
                            model = path_parts[proof_idx + 2]
                        if proof_idx + 3 < len(path_parts):
                            timestamp = path_parts[proof_idx + 3]

                    # Add metadata to the result
                    result_with_meta = {
                        **result,
                        "project": project,
                        "model": model,
                        "timestamp": timestamp,
                        "source_file": validity_path,
                    }
                    valid_proofs.append(result_with_meta)
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing {validity_path}: {e}")

    return valid_proofs


def count_lines(text: str) -> int:
    """Count non-empty lines in a string."""
    return len([line for line in text.strip().split("\n") if line.strip()])


def display_proof(proof: Dict[str, Any], verbose: bool = False) -> None:
    """Display a proof with metadata in a rich panel."""
    # Create a table for metadata
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Key", style="cyan")
    meta_table.add_column("Value", style="green")

    meta_table.add_row("Project", proof["project"])
    meta_table.add_row("Model", proof["model"])
    meta_table.add_row("Timestamp", proof["timestamp"])

    if "theorem_name" in proof:
        meta_table.add_row("Theorem", proof["theorem_name"])
    elif "problem_name" in proof:
        meta_table.add_row("Problem", proof["problem_name"])

    # Add statement if available
    if "statement" in proof and proof["statement"]:
        meta_table.add_row("Statement", proof["statement"])

    # Get the original proof code and the version without comments
    lean_code = proof.get("lean_proof", "")
    lean_code_no_comments = remove_lean_comments(lean_code)
    if not lean_code_no_comments:
        lean_code_no_comments = lean_code

    # Add proof length information
    line_count = count_lines(lean_code)
    char_count = len(lean_code)
    lines_no_comments = count_lines(lean_code_no_comments)
    chars_no_comments = len(lean_code_no_comments)

    meta_table.add_row("Original Length", f"{line_count} lines, {char_count} characters")
    meta_table.add_row("Without Comments", f"{lines_no_comments} lines, {chars_no_comments} characters")

    # Show the percentage of the proof that is comments
    if char_count > 0:
        comment_percentage = ((char_count - chars_no_comments) / char_count) * 100
        meta_table.add_row("Comments", f"{comment_percentage:.1f}% of the proof")

    # Show the Lean code in a syntax highlighted block
    code_block = Syntax(lean_code, "lean", theme="monokai", line_numbers=True, word_wrap=True)

    # Also show the code without comments
    code_block_no_comments = Syntax(lean_code_no_comments, "lean", theme="monokai", line_numbers=True, word_wrap=True)

    # Create the full display
    console.print(meta_table)
    console.print()
    console.print(Panel(code_block, title="Original Lean Proof", subtitle=f"[{line_count} lines]", expand=False))
    console.print()
    console.print(
        Panel(
            code_block_no_comments,
            title="Lean Proof (Comments Removed)",
            subtitle=f"[{lines_no_comments} lines]",
            expand=False,
        )
    )

    # In verbose mode, show the full proof object
    if verbose:
        console.print()
        console.print("Full proof object:")
        console.print(
            json.dumps(
                # Remove the source file and lean_proof to avoid duplication
                {k: v for k, v in proof.items() if k not in ["source_file", "lean_proof"]},
                indent=2,
            )
        )

    console.print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Display random valid proofs from evaluation runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "proof"),
        help="Directory containing evaluation results",
    )
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--project", type=str, default=None, help="Filter by project name")
    parser.add_argument("--min-lines", type=int, default=None, help="Minimum number of lines in proof")
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum number of lines in proof")
    parser.add_argument("--verbose", action="store_true", help="Show full proof objects")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--only-rechecked", action="store_true", help="Only include results from timestamps with '_recheck' suffix"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    console.print(f"[bold]Looking for validity results in: {args.results_dir}")

    # Find all validity_results.jsonl files
    validity_paths = find_validity_results(args.results_dir)
    console.print(f"[green]Found {len(validity_paths)} validity result files")

    if not validity_paths:
        console.print("[yellow]No validity result files found. Did you specify the correct directory?")
        return

    # Collect valid proofs from all files
    all_valid_proofs = []
    for path in tqdm(validity_paths, desc="Processing files"):
        valid_proofs = extract_valid_proofs(path)
        all_valid_proofs.extend(valid_proofs)

    console.print(f"[green]Found {len(all_valid_proofs)} valid proofs")

    # Apply filters
    filtered_proofs = all_valid_proofs

    if args.model:
        filtered_proofs = [p for p in filtered_proofs if args.model.lower() in p["model"].lower()]
        console.print(f"[blue]Filtered to {len(filtered_proofs)} proofs from model '{args.model}'")

    if args.project:
        filtered_proofs = [p for p in filtered_proofs if args.project.lower() in p["project"].lower()]
        console.print(f"[blue]Filtered to {len(filtered_proofs)} proofs from project '{args.project}'")

    if args.min_lines is not None:
        filtered_proofs = [p for p in filtered_proofs if count_lines(p.get("lean_proof", "")) >= args.min_lines]
        console.print(f"[blue]Filtered to {len(filtered_proofs)} proofs with at least {args.min_lines} lines")

    if args.max_lines is not None:
        filtered_proofs = [p for p in filtered_proofs if count_lines(p.get("lean_proof", "")) <= args.max_lines]
        console.print(f"[blue]Filtered to {len(filtered_proofs)} proofs with at most {args.max_lines} lines")

    if args.only_rechecked:
        filtered_proofs = [p for p in filtered_proofs if p["timestamp"].endswith("_recheck")]
        console.print(f"[blue]Filtered to {len(filtered_proofs)} proofs with '_recheck' in the timestamp")

    if not filtered_proofs:
        console.print("[yellow]No valid proofs match the specified filters")
        return

    # Select random proofs
    count = len(filtered_proofs)
    selected_proofs = random.sample(filtered_proofs, count)

    # Display the selected proofs one at a time, waiting for user input between each
    console.print(f"[bold green]Displaying {count} random proofs. Press Enter to see each proof:[/bold green]\n")

    for i, proof in enumerate(selected_proofs, 1):
        if i > 1:  # Don't wait before showing the first proof
            console.print("[italic]Press Enter to see the next proof...[/italic]")
            input()

        console.print(f"[bold cyan]Proof {i} of {count}")
        display_proof(proof, verbose=args.verbose)

    console.print("[bold green]All proofs displayed.[/bold green]")


if __name__ == "__main__":
    main()
