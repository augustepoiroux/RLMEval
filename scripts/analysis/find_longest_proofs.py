#!/usr/bin/env python3
"""
Find and display the longest Lean proofs from evaluation runs.
Extracts information from blueprint files and model-generated proofs.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import jsonlines
import networkx as nx
from lean_interact.utils import remove_lean_comments
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
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
class ProofInfo:
    """Information about a specific proof."""

    model: str
    timestamp: str
    project: str
    file_path: str
    length_chars: int
    length_lines: int
    proof_text: str
    theorem_name: str | None = None
    theorem_statement: str | None = None
    theorem_decl: str | None = None  # Full theorem declaration
    source_file: str | None = None
    ground_truth_proof: str | None = None
    informal_proof: str | None = None


def count_lines(text: str) -> int:
    """Count non-empty lines in a string."""
    if not text:
        return 0
    return len([line for line in text.strip().split("\n") if line.strip()])


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


def extract_theorem_from_path(path: str) -> str | None:
    """Extract theorem name from the folder path containing validity_results.jsonl."""
    # Find the parent folder of validity_results.jsonl, which should be the declaration name
    parent_dir = os.path.basename(os.path.dirname(path))
    if parent_dir and parent_dir != "":
        return parent_dir
    return None


def load_blueprint_data(project_dir: str) -> tuple[list[dict], dict, list[dict]] | None:
    """
    Load blueprint data and lean files from a project directory.
    Returns (blueprint_to_lean, lean_files, lean_declarations)
    """
    try:
        # Load blueprint to lean mapping
        blueprint_to_lean_path = os.path.join(project_dir, "blueprint_to_lean.jsonl")
        if not os.path.exists(blueprint_to_lean_path):
            console.print(f"[yellow]Warning: No blueprint_to_lean.jsonl found in {project_dir}")
            return None

        with jsonlines.open(blueprint_to_lean_path) as reader:
            blueprint_to_lean = list(reader)

        # Load lean files
        lean_files_path = os.path.join(project_dir, "lean_files.jsonl")
        if not os.path.exists(lean_files_path):
            console.print(f"[yellow]Warning: No lean_files.jsonl found in {project_dir}")
            return None

        with jsonlines.open(lean_files_path) as reader:
            lean_files_list = list(reader)
            lean_files = {file["file"]: file["content"] for file in lean_files_list}

        # Load lean declarations
        lean_declarations_path = os.path.join(project_dir, "lean_declarations.jsonl")
        if not os.path.exists(lean_declarations_path):
            console.print(f"[yellow]Warning: No lean_declarations.jsonl found in {project_dir}")
            return None

        with jsonlines.open(lean_declarations_path) as reader:
            lean_declarations = list(reader)

        return blueprint_to_lean, lean_files, lean_declarations

    except Exception as e:
        console.print(f"[red]Error loading blueprint data: {e}")
        return None


def extract_node_informal_text(node: dict[str, Any]) -> str | None:
    """Extract informal proof text from a node in the blueprint."""
    if "proof" not in node or "text" not in node["proof"]:
        return None

    processed_text = node["proof"]["text"]
    if isinstance(processed_text, str):
        return "\n".join(line.strip() for line in processed_text.split(r"\\") if line.strip())
    return None


def extract_info_from_blueprint(
    blueprint_graph: nx.DiGraph, lean_decls_map: dict | None, theorem_name: str
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """
    Extract information from blueprint graph and declarations.
    Returns (theorem_statement, ground_truth_proof, informal_proof, source_file, theorem_decl)
    """
    theorem_statement = None
    ground_truth_proof = None
    informal_proof = None
    source_file = None
    theorem_decl = None

    # Only match node label exactly to theorem_name
    if blueprint_graph and theorem_name in blueprint_graph.nodes:
        node = blueprint_graph.nodes[theorem_name]
        # Try to extract from lean_declarations in the node
        for lean_decl in node.get("lean_declarations", []):
            if not isinstance(lean_decl, dict):
                continue
            if "theorem_info" in lean_decl and "declsig" in lean_decl["theorem_info"]:
                theorem_statement = lean_decl["theorem_info"]["declsig"]
            elif "declsig" in lean_decl:
                theorem_statement = lean_decl["declsig"]
            if "file" in lean_decl:
                source_file = lean_decl["file"]
                if "theorem_info" in lean_decl and "proof" in lean_decl["theorem_info"]:
                    ground_truth_proof = lean_decl["theorem_info"]["proof"]
                    if ground_truth_proof.startswith(":= by"):
                        ground_truth_proof = ground_truth_proof[5:].strip()
                    ground_truth_proof = clean_proof(ground_truth_proof)
        informal_proof = extract_node_informal_text(node)
        return theorem_statement, ground_truth_proof, informal_proof, source_file, theorem_decl

    console.print(f"[yellow]Warning: No matching node found for theorem name: {theorem_name}")

    # Nothing found after all attempts
    return None, None, None, None, None


def find_valid_proofs(
    validity_path: str,
    no_comments: bool = False,
    blueprint_graph: nx.DiGraph | None = None,
    lean_decls_map: dict | None = None,
) -> list[ProofInfo]:
    """Find all valid proofs in a validity_results.jsonl file."""
    proofs = []

    model, timestamp, project = extract_path_info(validity_path)

    try:
        with jsonlines.open(validity_path) as reader:
            for result in reader:
                # In proof evaluation, the field is 'lean_proof'
                lean_proof = result.get("lean_proof")
                well_typed = result.get("well_typed", False)

                if lean_proof and well_typed:
                    # Remove comments if requested
                    if no_comments:
                        lean_proof_no_comments = clean_proof(lean_proof)
                        if lean_proof_no_comments:
                            lean_proof = lean_proof_no_comments

                    # Extract theorem name from path or result
                    theorem_name = extract_theorem_from_path(validity_path)

                    # Initialize with defaults
                    theorem_statement = None
                    theorem_decl = None
                    source_file = None
                    ground_truth_proof = None
                    informal_proof = None

                    # Use only blueprint data if available
                    if blueprint_graph is not None and theorem_name:
                        bp_statement, bp_proof, bp_informal, bp_source, bp_decl = extract_info_from_blueprint(
                            blueprint_graph, lean_decls_map, theorem_name
                        )

                        theorem_statement = bp_statement
                        ground_truth_proof = bp_proof
                        informal_proof = bp_informal
                        source_file = bp_source
                        theorem_decl = bp_decl

                    proof_info = ProofInfo(
                        model=model,
                        timestamp=timestamp,
                        project=project,
                        file_path=validity_path,
                        length_chars=len(lean_proof),
                        length_lines=count_lines(lean_proof),
                        proof_text=lean_proof,
                        theorem_name=theorem_name,
                        theorem_statement=theorem_statement,
                        theorem_decl=theorem_decl,
                        source_file=source_file,
                        ground_truth_proof=ground_truth_proof,
                        informal_proof=informal_proof,
                    )
                    proofs.append(proof_info)
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing {validity_path}: {e}")

    return proofs


def find_validity_results(results_dir: str) -> list[str]:
    """Find all validity_results.jsonl files in the given directory."""
    return glob.glob(f"{results_dir}/**/validity_results.jsonl", recursive=True)


def find_all_proofs(
    results_dir: str, no_comments: bool = False, traced_repos_dir: str | None = None, only_rechecked: bool = False
) -> list[ProofInfo]:
    """Find all valid proofs in the results directory."""
    console.print(f"[bold]Looking for validity results in: {results_dir}")

    # Load blueprint data if traced_repos_dir is provided
    blueprint_graphs = {}
    lean_decls_maps = {}  # Map from project to theorem name -> declaration mapping

    if traced_repos_dir:
        console.print(f"[bold]Loading blueprint data from: {traced_repos_dir}")
        project_dirs = [d for d in os.listdir(traced_repos_dir) if os.path.isdir(os.path.join(traced_repos_dir, d))]

        for project_dir in tqdm(project_dirs, desc="Loading blueprint data"):
            full_project_dir = os.path.join(traced_repos_dir, project_dir)
            blueprint_data = load_blueprint_data(full_project_dir)

            if blueprint_data:
                blueprint_to_lean, lean_files, lean_declarations = blueprint_data

                # Build the DAG of the blueprint
                blueprint_graph = nx.DiGraph()

                for node in blueprint_to_lean:
                    if not node["label"]:
                        continue
                    blueprint_graph.add_node(node["label"], **node)
                    for use in node.get("uses", []):
                        blueprint_graph.add_edge(use, node["label"])

                # Build a mapping from theorem names to declarations for fast lookup
                lean_decls_map = {}
                for decl in lean_declarations:
                    if "name" in decl:
                        lean_decls_map[decl["name"]] = decl

                project_key = project_dir.split("_")[0]
                blueprint_graphs[project_key] = blueprint_graph
                lean_decls_maps[project_key] = lean_decls_map

        console.print(f"[green]Loaded blueprint data for {len(blueprint_graphs)} projects")

    # Find all validity_results.jsonl files
    validity_paths = find_validity_results(results_dir)
    console.print(f"[green]Found {len(validity_paths)} validity result files")

    all_proofs = []
    for path in tqdm(validity_paths, desc="Processing files"):
        # Extract timestamp and check if we should skip based on only_rechecked flag
        _, timestamp, _ = extract_path_info(path)
        if only_rechecked and "_recheck" not in timestamp:
            continue

        # Determine which blueprint graph to use based on project name
        blueprint_graph = None
        lean_decls_map = None
        _, _, project = extract_path_info(path)

        for project_key in blueprint_graphs:
            if project_key == project:
                blueprint_graph = blueprint_graphs[project_key]
                lean_decls_map = lean_decls_maps[project_key]
                break

        proofs = find_valid_proofs(
            path, no_comments=no_comments, blueprint_graph=blueprint_graph, lean_decls_map=lean_decls_map
        )
        all_proofs.extend(proofs)

    if only_rechecked:
        console.print("[blue]Filtered to only include results from '_recheck' timestamps")

    console.print(f"[green]Found {len(all_proofs)} valid proofs")
    return all_proofs


def find_longest_proofs(
    all_proofs: list[ProofInfo], num_proofs: int = 10, by_chars: bool = True, by_model: bool = False
) -> dict[str, list[ProofInfo]]:
    """Find the longest proofs by characters or lines."""
    metric = "length_chars" if by_chars else "length_lines"

    if by_model:
        # Group by model and find longest proofs for each model
        model_proofs = defaultdict(list)
        for proof in all_proofs:
            model_proofs[proof.model].append(proof)

        result = {}
        for model, proofs in model_proofs.items():
            result[model] = sorted(proofs, key=lambda p: getattr(p, metric), reverse=True)[:num_proofs]
        return result
    else:
        # Find longest proofs across all models
        return {"all": sorted(all_proofs, key=lambda p: getattr(p, metric), reverse=True)[:num_proofs]}


def print_proof_table(longest_proofs: dict[str, list[ProofInfo]], by_chars: bool = True):
    """Print a table of the longest proofs."""
    metric_name = "Characters" if by_chars else "Lines"

    for model, proofs in longest_proofs.items():
        table = Table(title=f"Longest Proofs by {metric_name} for {model}")

        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Theorem", style="magenta")
        table.add_column(metric_name, style="green", justify="right")
        table.add_column("Model", style="blue")
        table.add_column("Project", style="yellow")
        table.add_column("Source File", style="dim")

        for i, proof in enumerate(proofs, 1):
            metric_value = proof.length_chars if by_chars else proof.length_lines
            theorem_name = proof.theorem_name or "Unknown"
            source_file = proof.source_file or "Unknown"

            # Shorten the source file path for display
            if len(source_file) > 30:
                source_file = "..." + source_file[-27:]

            table.add_row(str(i), theorem_name, str(metric_value), proof.model, proof.project, source_file)

        console.print(table)
        console.print()


def print_full_proof(
    proof: ProofInfo,
    show_statement: bool = True,
    show_ground_truth: bool = True,
    show_informal: bool = True,
    show_decl: bool = True,
):
    """Print the full text of a proof with additional information."""
    theorem_name = proof.theorem_name or "Unknown Theorem"
    title = f"Longest Proof: {theorem_name} ({proof.length_lines} lines, {proof.length_chars} chars)"

    # Print full theorem declaration if available
    if show_decl and proof.theorem_decl:
        console.print(
            Panel(
                Syntax(proof.theorem_decl, "lean4", theme="monokai"),
                title=f"Full Theorem Declaration: {theorem_name}",
                subtitle=f"Source: {proof.source_file or 'Unknown'}",
            )
        )
        console.print()
    # Print theorem statement if available and declaration not shown
    elif show_statement and proof.theorem_statement:
        console.print(
            Panel(
                Syntax(proof.theorem_statement, "lean4", theme="monokai"),
                title=f"Theorem Statement: {theorem_name}",
                subtitle=f"Source: {proof.source_file or 'Unknown'}",
            )
        )
        console.print()

    # Print informal proof if available
    if show_informal and proof.informal_proof:
        console.print(Panel(proof.informal_proof, title="Informal Proof", subtitle="Natural language description"))
        console.print()

    # Print model-generated proof
    console.print(
        Panel(
            Syntax(proof.proof_text, "lean4", theme="monokai", line_numbers=True),
            title=title,
            subtitle=f"Model: {proof.model}, Project: {proof.project}",
        )
    )

    # Print ground truth proof if available
    if show_ground_truth and proof.ground_truth_proof:
        gt_lines = count_lines(proof.ground_truth_proof)
        console.print()
        console.print(
            Panel(
                Syntax(proof.ground_truth_proof, "lean4", theme="monokai", line_numbers=True),
                title=f"Ground Truth Proof: {theorem_name} ({gt_lines} lines)",
                subtitle="Reference implementation",
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Find the longest proofs from evaluation runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "proof"),
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--traced-repos-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "traced_repos"),
        help="Directory containing traced repositories with blueprint data",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save the results JSON (optional)")
    parser.add_argument("--model", type=str, default=None, help="Only find proofs for a specific model")
    parser.add_argument("--num-proofs", type=int, default=10, help="Number of longest proofs to display")
    parser.add_argument("--by-lines", action="store_true", help="Sort by number of lines instead of characters")
    parser.add_argument("--by-model", action="store_true", help="Group longest proofs by model")
    parser.add_argument("--print-proof", action="store_true", help="Print the full text of the longest proof")
    parser.add_argument("--print-top-n", type=int, default=0, help="Print the full text of the top N longest proofs")
    parser.add_argument("--hide-statement", action="store_true", help="Don't show theorem statements")
    parser.add_argument("--hide-ground-truth", action="store_true", help="Don't show ground truth proofs")
    parser.add_argument("--hide-informal", action="store_true", help="Don't show informal proofs")
    parser.add_argument("--hide-decl", action="store_true", help="Don't show full theorem declarations")
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Remove comments from proofs before calculating length and displaying",
    )
    parser.add_argument(
        "--no-blueprint", action="store_true", help="Don't use blueprint files (faster but less information)"
    )
    parser.add_argument(
        "--only-rechecked", action="store_true", help="Only include results from timestamps with '_recheck' suffix"
    )
    args = parser.parse_args()

    # Find all proofs
    traced_repos_dir = None if args.no_blueprint else args.traced_repos_dir
    all_proofs = find_all_proofs(
        args.results_dir,
        no_comments=args.no_comments,
        traced_repos_dir=traced_repos_dir,
        only_rechecked=args.only_rechecked,
    )

    if args.model:
        all_proofs = [p for p in all_proofs if p.model == args.model]
        console.print(f"[green]Filtered to {len(all_proofs)} proofs for model {args.model}")

    # Find longest proofs
    by_chars = not args.by_lines
    longest_proofs = find_longest_proofs(
        all_proofs, num_proofs=args.num_proofs, by_chars=by_chars, by_model=args.by_model
    )

    # Print table of longest proofs
    print_proof_table(longest_proofs, by_chars=by_chars)

    # Print full text of the longest proof if requested
    if args.print_proof and longest_proofs:
        # If by-model is also set, print the longest proof from each model
        if args.by_model:
            for model, proofs in longest_proofs.items():
                if proofs:
                    console.print(f"\n[bold cyan]Longest proof for model: {model}[/bold cyan]\n")
                    print_full_proof(
                        proofs[0],
                        show_statement=not args.hide_statement,
                        show_ground_truth=not args.hide_ground_truth,
                        show_informal=not args.hide_informal,
                        show_decl=not args.hide_decl,
                    )
                    console.print("\n" + "=" * 80 + "\n")  # Separator between proofs
        else:
            # Print the overall longest proof
            all_longest = []
            for proofs in longest_proofs.values():
                all_longest.extend(proofs)

            # Sort all longest proofs
            metric = "length_chars" if by_chars else "length_lines"
            all_longest = sorted(all_longest, key=lambda p: getattr(p, metric), reverse=True)

            if all_longest:
                print_full_proof(
                    all_longest[0],
                    show_statement=not args.hide_statement,
                    show_ground_truth=not args.hide_ground_truth,
                    show_informal=not args.hide_informal,
                    show_decl=not args.hide_decl,
                )

    # Print full text of top N longest proofs if requested
    if args.print_top_n > 0 and longest_proofs:
        # If by-model is also set, we want to show top N for each model
        if args.by_model:
            for model, proofs in longest_proofs.items():
                console.print(
                    f"\n[bold cyan]Top {min(args.print_top_n, len(proofs))} proofs for model: {model}[/bold cyan]\n"
                )
                for i, proof in enumerate(proofs[: args.print_top_n]):
                    if i > 0:
                        console.print("\n" + "-" * 50 + "\n")  # Lighter separator between proofs of same model
                    print_full_proof(
                        proof,
                        show_statement=not args.hide_statement,
                        show_ground_truth=not args.hide_ground_truth,
                        show_informal=not args.hide_informal,
                        show_decl=not args.hide_decl,
                    )
                console.print("\n" + "=" * 80 + "\n")  # Heavier separator between models
        else:
            # Print top N across all models
            all_longest = []
            for proofs in longest_proofs.values():
                all_longest.extend(proofs)

            metric = "length_chars" if by_chars else "length_lines"
            all_longest = sorted(all_longest, key=lambda p: getattr(p, metric), reverse=True)

            for i, proof in enumerate(all_longest[: args.print_top_n]):
                if i > 0:
                    console.print("\n" + "=" * 80 + "\n")  # Separator between proofs
                print_full_proof(
                    proof,
                    show_statement=not args.hide_statement,
                    show_ground_truth=not args.hide_ground_truth,
                    show_informal=not args.hide_informal,
                    show_decl=not args.hide_decl,
                )

    # Save results to JSON if requested
    if args.output:
        # Convert to serializable format
        serializable = {}
        for model, proofs in longest_proofs.items():
            serializable[model] = [
                {
                    "model": p.model,
                    "timestamp": p.timestamp,
                    "project": p.project,
                    "file_path": p.file_path,
                    "length_chars": p.length_chars,
                    "length_lines": p.length_lines,
                    "theorem_name": p.theorem_name,
                    "theorem_statement": p.theorem_statement,
                    "theorem_decl": p.theorem_decl,
                    "source_file": p.source_file,
                    "proof_text": p.proof_text,
                    "ground_truth_proof": p.ground_truth_proof,
                    "informal_proof": p.informal_proof,
                }
                for p in proofs
            ]

        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        console.print(f"[green]Results saved to: {args.output}")


a = """
  intro h
  replace h := mul_dvd_mul_left (η - 1) h
  rw [← z_spec] at h
  rw [← pow_two] at h
  exact lambda_sq_not_dvd_a_add_eta_sq_mul_b _ h/-- Let `a`, `b`, `c` be in `ℕ`. If `3 ∣ a` and `3 ∣ c` and `a ^ 3 + b ^ 3 = c ^ 3`,
then 3 divides the `gcd` of the set `{a, b, c}`. -/
lemma three_dvd_gcd_of_dvd_a_of_dvd_c {a b c : ℕ} (ha : 3 ∣ a) (hc : 3 ∣ c)
    (hF : a ^ 3 + b ^ 3 = c ^ 3) : 3 ∣ ({a, b, c} : Finset ℕ).gcd id := by
  have hb : 3 ∣ b := by
    have : 3 ∣ (b : ℤ) ^ 3 := by
      replace hF : (a : ℤ) ^ 3 + (b : ℤ) ^ 3 = (c : ℤ) ^ 3 := by exact_mod_cast hF
      rw [add_comm, ← eq_sub_iff_add_eq] at hF
      rw [hF]
      exact dvd_sub (dvd_pow (by exact_mod_cast hc) (by decide))
        (dvd_pow (by exact_mod_cast ha) (by decide))
    exact Int.coe_nat_dvd.1 <| Int.prime_three.dvd_of_dvd_pow this
  refine Finset.dvd_gcd (fun x hx ↦ ?_)
  simp only [Finset.mem_insert, Finset.mem_singleton] at hx
  rcases hx with (hx | hx | hx)
  · exact hx ▸ ha
  · exact hx ▸ hb
  · exact hx ▸ hc
"""

if __name__ == "__main__":
    # print(clean_proof(a))
    main()
