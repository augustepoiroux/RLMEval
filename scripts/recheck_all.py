import argparse
import os
import subprocess
from datetime import datetime
from glob import glob

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rlm_eval.utils import ROOT_DIR, logger

console = Console()


def find_all_summary_timestamps():
    """Find all timestamps in the results/proof/summary folder."""
    summary_dir = os.path.join(ROOT_DIR, "results", "proof", "summary")
    if not os.path.exists(summary_dir):
        console.print("[bold red]Summary directory not found![/bold red]")
        return []

    results = []
    for model_dir in os.listdir(summary_dir):
        model_path = os.path.join(summary_dir, model_dir)
        if os.path.isdir(model_path):
            for timestamp_dir in os.listdir(model_path):
                timestamp_path = os.path.join(model_path, timestamp_dir)

                # Check if this is a valid timestamp directory with configs
                benchmark_config_path = os.path.join(timestamp_path, "benchmark_config.yaml")
                model_config_path = os.path.join(timestamp_path, "model_config.yaml")

                if os.path.exists(benchmark_config_path) and os.path.exists(model_config_path):
                    # Try to load configs to verify they're valid
                    try:
                        with open(benchmark_config_path, "r") as f:
                            benchmark_config = yaml.safe_load(f)
                        with open(model_config_path, "r") as f:
                            model_config = yaml.safe_load(f)

                        results.append(
                            {
                                "model": model_dir,
                                "timestamp": timestamp_dir,
                                "model_config_path": model_config_path,
                                "benchmark_config_path": benchmark_config_path,
                                "model_config": model_config,
                                "benchmark_config": benchmark_config,
                            }
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error loading configs from {timestamp_path}: {e}[/yellow]")

    return results


def recheck_timestamp(timestamp_info, output_timestamp=None, log_dir=None):
    """Rerun the checking for a specific timestamp."""
    model_config_path = timestamp_info["model_config_path"]
    benchmark_config_path = timestamp_info["benchmark_config_path"]
    timestamp = timestamp_info["timestamp"]
    model_name = timestamp_info["model"]

    # Create a unique output timestamp if not provided
    if output_timestamp is None:
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_recheck"

    # Create log directory if specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{model_name}_{timestamp}_recheck.log"
        log_path = os.path.join(log_dir, log_filename)
    else:
        log_path = None

    cmd = [
        "python",
        "scripts/eval_proof_autoformalization.py",
        "--benchmark-config",
        benchmark_config_path,
        "--model-config",
        model_config_path,
        "--recheck-timestamp",
        timestamp,
    ]

    console.print(
        Panel.fit(
            f"Running recheck for [bold cyan]{timestamp_info['model']}[/bold cyan] timestamp [bold green]{timestamp}[/bold green]",
            title="Recheck Started",
        )
    )

    try:
        # Run the command and capture output
        if log_path:
            # If log file specified, save output to file
            with open(log_path, "w") as log_file:
                # Write header information
                log_file.write(f"=== Recheck Log for {model_name} - {timestamp} ===\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 60 + "\n\n")

                # Run command and tee output to both log file and capture
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                output = []
                if process.stdout:
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()  # Ensure immediate writing to file
                        output.append(line)

                returncode = process.wait()

                # Write footer
                log_file.write("\n" + "=" * 60 + "\n")
                log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Return code: {returncode}\n")

            console.print(f"[bold blue]Log saved to: {log_path}[/bold blue]")

            if returncode == 0:
                console.print(
                    f"[bold green]Successfully rechecked {timestamp_info['model']} - {timestamp}[/bold green]"
                )
                return True
            else:
                console.print(f"[bold red]Failed to recheck {timestamp_info['model']} - {timestamp}[/bold red]")
                console.print(f"[red]Check log file for details: {log_path}[/red]")
                return False

        else:
            # Original behavior without logging
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                console.print(
                    f"[bold green]Successfully rechecked {timestamp_info['model']} - {timestamp}[/bold green]"
                )
                return True
            else:
                console.print(f"[bold red]Failed to recheck {timestamp_info['model']} - {timestamp}[/bold red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
    except Exception as e:
        console.print(f"[bold red]Error executing recheck command: {e}[/bold red]")
        if log_path:
            console.print(f"[red]See log file for details: {log_path}[/red]")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun checks for all timestamps in the results/proof/summary folder")
    parser.add_argument("--model", type=str, help="Specific model to recheck (e.g., 'gpt-4o-2024-05-13')")
    parser.add_argument("--timestamp", type=str, help="Specific timestamp to recheck")
    parser.add_argument("--limit", type=int, help="Limit the number of timestamps to process")
    parser.add_argument("--output-timestamp", type=str, help="Custom timestamp suffix for the recheck output folders")
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to save recheck logs (default: results/proof/recheck_logs)",
        default="results/proof/recheck_logs",
    )
    args = parser.parse_args()

    # Find all timestamps
    console.print("[bold]Searching for timestamp directories...[/bold]")
    all_timestamps = find_all_summary_timestamps()

    # Filter based on command line arguments
    if args.model:
        all_timestamps = [t for t in all_timestamps if t["model"] == args.model]

    if args.timestamp:
        all_timestamps = [t for t in all_timestamps if t["timestamp"] == args.timestamp]

    if args.limit and args.limit > 0:
        all_timestamps = all_timestamps[: args.limit]

    if not all_timestamps:
        console.print("[yellow]No matching timestamp directories found.[/yellow]")
        exit(0)

    # Display what will be processed
    console.print(f"[bold]Found {len(all_timestamps)} timestamp(s) to recheck:[/bold]")
    for i, t in enumerate(all_timestamps, 1):
        console.print(f"  {i}. [cyan]{t['model']}[/cyan] - [green]{t['timestamp']}[/green]")

    # Get confirmation
    if input("\nProceed with rechecking? (y/n): ").lower() != "y":
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        exit(0)

    # Process all timestamps
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[bold]Rechecking timestamps...", total=len(all_timestamps))

        for idx, timestamp_info in enumerate(all_timestamps):
            progress.update(
                task, description=f"[bold]Rechecking {timestamp_info['model']} - {timestamp_info['timestamp']}"
            )

            success = recheck_timestamp(timestamp_info, args.output_timestamp, args.log_dir)
            results.append(
                {"model": timestamp_info["model"], "timestamp": timestamp_info["timestamp"], "success": success}
            )

            progress.update(task, advance=1)

    # Print summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    console.print("\n[bold]Recheck Summary:[/bold]")
    console.print(f"  [green]Successfully rechecked: {len(successful)}/{len(results)}[/green]")
    console.print(f"  [red]Failed rechecks: {len(failed)}/{len(results)}[/red]")

    if failed:
        console.print("\n[bold red]Failed rechecks:[/bold red]")
        for f in failed:
            console.print(f"  - {f['model']} - {f['timestamp']}")

    console.print("\n[bold green]Recheck operation completed![/bold green]")
