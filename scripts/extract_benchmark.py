import argparse
import json
import os
import time

import jsonlines
import yaml

from rlm_eval.data_processing.blueprint_extractor import (
    extract_blueprint_info,
)
from rlm_eval.data_processing.blueprint_lean_linking import (
    merge_blueprint_lean_dep_graphs,
)
from rlm_eval.data_processing.lean_project_extractor import trace_repo
from rlm_eval.utils import ROOT_DIR, console

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract benchmark info")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    repos = config.get("repositories", [])

    traced_repos_dir = os.path.join(ROOT_DIR, "traced_repos")
    success = []

    # Iterate over the repositories loaded from the YAML config
    for repo in repos:
        git_url = repo["git_url"]
        commit = repo["commit"]
        project_name = repo["project_name"]
        blueprint_cmd = repo.get("blueprint_cmd")

        try:
            start_time = time.time()
            project_dir = git_url.split("/")[-1]
            output_dir = os.path.join(traced_repos_dir, f"{project_dir}_{commit}")
            _, declarations = trace_repo(output_dir, git_url, commit, build_deps=False)

            project_root_dir = os.path.join(output_dir, project_dir)

            if blueprint_cmd:
                console.print(f"Running command: `{blueprint_cmd}`")
                os.chdir(project_root_dir)
                os.system(blueprint_cmd)

            blueprint_src_path = os.path.join(project_root_dir, "blueprint")
            blueprint_graph = extract_blueprint_info(blueprint_src_path)

            with jsonlines.open(os.path.join(output_dir, "blueprint.jsonl"), "w") as writer:
                writer.write_all(blueprint_graph)

            console.print(f"Number of declarations in blueprint: {len(blueprint_graph)}")
            console.print(f"Number of declarations in Lean: {len(declarations)}")

            dep_graph_info = merge_blueprint_lean_dep_graphs(blueprint_graph, declarations)

            with jsonlines.open(os.path.join(output_dir, "blueprint_to_lean.jsonl"), "w") as writer:
                writer.write_all(dep_graph_info)

            success.append([project_name, "Success"])
            console.print(
                f"Successfully processed {project_name} at commit {commit} in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            console.print(f"Failed to process {project_name} at commit {commit}")
            console.print_exception(width=None)
            success.append([project_name, "Failure: " + str(e)])

    console.print(success)
    with open(os.path.join(traced_repos_dir, "success.json"), "w") as f:
        json.dump(success, f)
