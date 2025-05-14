import argparse
import concurrent.futures
import dataclasses
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import jsonlines
import litellm
import networkx as nx
import yaml
from lean_interact import AutoLeanServer, LeanREPLConfig, LocalProject
from lean_interact.interface import Command, LeanError
from lean_interact.utils import clean_theorem_string
from litellm import completion, text_completion
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.exceptions import ContextWindowExceededError
from litellm.utils import token_counter
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from tqdm import tqdm

from rlm_eval.data_processing.lean_utils import LeanFilesProcessor, trim_comments_end
from rlm_eval.utils import (
    DATA_DIR,
    ROOT_DIR,
    clean_messages,
    console,
    extract_lean_codes,
    generate_n_samples_sequence,
    logger,
    optional_status,
)

litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=os.path.join(ROOT_DIR, ".cache", "litellm"))


class PromptContext(Enum):
    NO_CONTEXT = "no_context"
    FILE_CONTEXT = "file_context"
    FILE_CONTEXT_NO_LEMMAS = "file_context_no_lemmas"


class ProvingEvaluation:
    def __init__(
        self,
        blueprint_with_lean: list[dict],
        lean_files: dict,
        lean_declarations: list[dict],
        project_dir: str,
    ):
        self.blueprint_with_lean = blueprint_with_lean
        self.lean_files = lean_files
        self.lean_declarations = lean_declarations
        self.project_dir = project_dir

        # preloading the Lean server to make sure it is cached before using multiprocessing
        self.repl_config = LeanREPLConfig(project=LocalProject(project_dir))

    def run(
        self,
        output_folder: str,
        model: str,
        use_chat_prompt: bool = False,
        api_key: str | None = None,
        api_base_url: str | None = None,
        max_total_tokens: int = 4096,
        max_generated_tokens: int = 512,
        nb_attempts: int = 1,
        temperature: float = 1.0,
        stopwords: list[str] = [],
        top_p: float = 0.95,
        verbose: bool = False,
        n_processes: int | None = 1,
        prompt_context: PromptContext = PromptContext.FILE_CONTEXT,
        gen_processes: int | None = None,
    ) -> tuple[int, int]:
        # first build the DAG of the blueprint
        blueprint_graph = nx.DiGraph()

        non_null_labels = [node["label"] for node in self.blueprint_with_lean if node["label"]]
        assert len(non_null_labels) == len(set(non_null_labels)), "Duplicate labels in the blueprint"

        for node in self.blueprint_with_lean:
            if not node["label"]:
                continue
            blueprint_graph.add_node(node["label"], **node)
            for use in node.get("uses", []):
                blueprint_graph.add_edge(use, node["label"])

        total_input_tokens, total_output_tokens = 0, 0

        # instantiate list of statements that are eligible for proving
        node_labels = list(reversed(list(nx.topological_sort(blueprint_graph))))
        eligible_node_labels = []
        for node_label in node_labels:
            node = blueprint_graph.nodes[node_label]
            if "lean_declarations" not in node:
                logger.info(f"Skipping {node_label}: no Lean declarations found.")
                continue
            if "proof" not in node or "text" not in node["proof"]:
                logger.info(f"Skipping {node_label}: no informal proof found.")
                continue
            lean_declarations_with_file = [lean_decl for lean_decl in node["lean_declarations"] if "file" in lean_decl]
            if len(lean_declarations_with_file) > 1:
                logger.info(
                    f"Skipping {node_label}: multiple Lean declarations per node is not yet supported for evaluation."
                )
                continue
            elif not lean_declarations_with_file:
                logger.info(f"Skipping {node_label}: no ground truth Lean declaration found.")
                continue
            eligible_node_labels.append(node_label)

        # eligible_node_labels = eligible_node_labels[:5]
        logger.info(f"Proving {len(eligible_node_labels)} theorems")

        # Function to prove a single node
        def prove_node_wrapper(
            node_label,
            blueprint_graph=blueprint_graph,
            lean_files=self.lean_files,
            lean_declarations=self.lean_declarations,
            output_folder=output_folder,
            verbose=verbose,
            nb_attempts=nb_attempts,
            temperature=temperature,
            top_p=top_p,
            model=model,
            use_chat_prompt=use_chat_prompt,
            stopwords=stopwords,
            api_key=api_key,
            api_base_url=api_base_url,
            max_generated_tokens=max_generated_tokens,
            max_total_tokens=max_total_tokens,
            prompt_context=prompt_context,
            console=console,
        ):
            node = blueprint_graph.nodes[node_label]
            try:
                predictions, input_tokens, output_tokens = _prove_node(
                    node=node,
                    blueprint_graph=blueprint_graph,
                    lean_files=lean_files,
                    lean_declarations=lean_declarations,
                    output_folder=os.path.join(output_folder, node_label),
                    verbose=verbose,
                    nb_attempts=nb_attempts,
                    temperature=temperature,
                    top_p=top_p,
                    model=model,
                    use_chat_prompt=use_chat_prompt,
                    stopwords=stopwords,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    max_generated_tokens=max_generated_tokens,
                    max_total_tokens=max_total_tokens,
                    prompt_context=prompt_context,
                    console=console,
                )
                return node, predictions, input_tokens, output_tokens, None
            except Exception as e:
                return node, None, 0, 0, str(e)

        results = []

        if gen_processes is not None and gen_processes > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=gen_processes) as executor:
                logger.info(f"Using {gen_processes} threads for proof generation")
                futures = [
                    executor.submit(prove_node_wrapper, node_label, verbose=False)
                    for node_label in eligible_node_labels
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Proving theorems"
                ):
                    if future.exception() is not None:
                        logger.error(f"Error while proving a node: {future.exception()}")
                    else:
                        results.append(future.result())
        else:
            # Standard sequential processing
            results = [
                prove_node_wrapper(node_label) for node_label in tqdm(eligible_node_labels, desc="Proving theorems")
            ]

        # Process results
        predictions_to_check = []
        total_input_tokens = 0
        total_output_tokens = 0
        for node, predictions, input_tokens, output_tokens, error in results:  # type: ignore
            if error:
                logger.error(f"Error while formalizing node {node['label']}: {error}")
                continue
            predictions_to_check.append((node, predictions))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        if verbose:
            console.print(f"Total input tokens: {total_input_tokens}")
            console.print(f"Total output tokens: {total_output_tokens}")

        self._check_predictions(
            blueprint_graph=blueprint_graph,
            predictions_to_check=predictions_to_check,
            output_folder=output_folder,
            nb_attempts=nb_attempts,
            verbose=verbose,
            nb_processes=n_processes,
        )

        return total_input_tokens, total_output_tokens

    def _check_predictions(
        self,
        blueprint_graph: nx.DiGraph,
        predictions_to_check: list[tuple[dict, list[str | None]]],
        output_folder: str,
        nb_attempts: int,
        verbose: bool,
        nb_processes: int | None = 1,
    ) -> None:
        # Prepare arguments for _process_predictions
        args_list = []
        for node, predictions in predictions_to_check:
            lean_declaration = node["lean_declarations"][0]
            original_file_content = self.lean_files[lean_declaration["file"]]
            args_list.append(
                ProcessPredictionsInput(
                    predictions=predictions,
                    lean_declaration=lean_declaration,
                    original_file_content=original_file_content,
                    node=node,
                    repl_config=self.repl_config,
                    timeout_context=360,
                    timeout_per_prediction=60,
                    output_folder=os.path.join(output_folder, node["label"]),
                )
            )

        # Define iterator
        executor = None
        if nb_processes is None or nb_processes > 1:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_processes)
            futures = [executor.submit(_process_predictions, args) for args in args_list]
            iterator = (
                future.result()
                if not future.exception()
                else (None, "", [PredictionEvaluationResult(lean_proof=None, error=str(future.exception()))])
                for future in concurrent.futures.as_completed(futures)
            )
        else:
            # Use simple loop
            iterator = map(_process_predictions, args_list)

        if verbose:
            iterator = tqdm(iterator, total=len(args_list), desc="Processing predictions")

        aggregated_check_results = defaultdict(lambda: defaultdict(float))
        pass_n_seq = generate_n_samples_sequence(nb_attempts, sequence_type="pow2")

        # Initialize counters for averages
        total_nodes = 0
        totals = defaultdict(int)
        totals |= {
            "Well-typed": 0,
            "System errors": 0,
            "Empty predictions": 0,
        }
        errors_collection = []
        for node_label, decl, validity_results in iterator:
            if node_label is None:
                # it means that an exception was raised
                errors_collection.append((node_label, validity_results[0].error))
                continue

            node = blueprint_graph.nodes[node_label]

            # Update counters
            total_nodes += 1

            # dump validity results to a file
            with jsonlines.open(os.path.join(output_folder, node_label, "validity_results.jsonl"), "w") as results_file:
                results_file.write_all(dataclasses.asdict(result) for result in validity_results)

            if verbose:
                console.rule(f"Formalization result for node {node_label}")
                console.print(
                    Panel(Syntax(node["processed_text"], "latex", word_wrap=True), title="LaTeX code to formalize")
                )
                console.print(
                    Panel(
                        Syntax(decl, "lean4", word_wrap=True, line_numbers=True),
                        title="Ground truth",
                    )
                )

            lean_codes = [result.lean_proof for result in validity_results if result.lean_proof is not None]
            well_typed_lean_codes: list[str] = [result.lean_proof for result in validity_results if result.well_typed]  # type: ignore
            assert all(code is not None for code in well_typed_lean_codes)

            check_results = defaultdict(lambda: defaultdict(float))

            def update_dict(key, result: PredictionEvaluationResult, target_dict: dict) -> None:
                target_dict["Well-typed"][key] = max(target_dict["Well-typed"].get(key, 0), float(result.well_typed))

            assert len(validity_results) == nb_attempts
            pass_n_seq_iter = iter(pass_n_seq)
            next_n = next(pass_n_seq_iter)
            cumulative_results = defaultdict(lambda: defaultdict(float))
            for idx_res, result in enumerate(validity_results):
                # Update cumulative results
                update_dict(None, result, cumulative_results)
                for key in cumulative_results:
                    check_results[key][next_n] = cumulative_results[key][None]

                if idx_res + 1 == next_n and next_n != nb_attempts:
                    next_n = next(pass_n_seq_iter)
            assert next(pass_n_seq_iter, None) is None
            assert next_n == nb_attempts

            # aggregate the results
            for key in set(aggregated_check_results.keys()).union(check_results.keys()):
                total = {"Well-typed": total_nodes}[key]
                aggregated_check_results[key]["Total"] = total
                if key not in check_results:
                    continue
                for n_samples, res in check_results[key].items():
                    aggregated_check_results[key][n_samples] += res

            # print stats so far with and without percentage
            table = Table(title="Proof stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Total", style="green")
            table.add_column("Percentage", style="green")
            for key in aggregated_check_results:
                table.add_row(key, "", "")
                total = int(aggregated_check_results[key]["Total"])
                for n_sample in pass_n_seq:
                    count = aggregated_check_results[key][n_sample]
                    percentage = (count / total) if total > 0 else 0
                    table.add_row(
                        f"  - pass@{n_sample}" if isinstance(n_sample, int) else f"  - {n_sample}",
                        f"{int(count)}/{total}",
                        f"{percentage:.2%}",
                    )
                table.add_section()
            console.print(table)

            # Collect counts for averages
            for result in validity_results:
                if result.well_typed:
                    totals["Well-typed"] += 1
                if result.error:
                    totals["System errors"] += 1
                    errors_collection.append((node_label, result.error))
                if result.lean_proof is None:
                    totals["Empty predictions"] += 1

            # Print averages
            table = Table(title="Averages on predictions per node")
            table.add_column("Metric", style="cyan")
            table.add_column("Total", style="green")
            table.add_column("Percentage", style="green")
            for key in totals:
                total = total_nodes * nb_attempts
                count = totals[key]
                percentage = (count / total) if total > 0 else 0
                table.add_row(key, f"{count}/{total}", f"{percentage:.2%}")
            console.print(table)

            # dump the aggregated results to a file
            with open(os.path.join(output_folder, "aggregated_results.json"), "w") as results_file:
                results_file.write(json.dumps(aggregated_check_results, indent=4))

            # dump the total stats to a file
            with open(os.path.join(output_folder, "total_stats.json"), "w") as results_file:
                results_file.write(json.dumps(totals, indent=4))

            # dump the errors to a file
            with jsonlines.open(os.path.join(output_folder, "errors.jsonl"), "w") as errors_file:
                errors_file.write_all({"node": node, "error": error} for node, error in errors_collection)

        if executor is not None:
            executor.shutdown(wait=False)


def _node_informal_text(node: dict[str, Any]) -> str:
    processed_text = node["proof"]["text"]
    return "\n".join(line.strip() for line in processed_text.split(r"\\") if line.strip())


def _node_informal_comment(node: dict[str, Any]) -> str:
    processed_text = node["proof"]["text"]
    processed_text = "\n".join(line.strip() for line in processed_text.split(r"\\") if line.strip())
    return f"/- {processed_text}\n-/"


def _prove_node(
    node: dict,
    blueprint_graph: nx.DiGraph,
    lean_files: dict,
    lean_declarations: list[dict],
    output_folder: str,
    verbose: bool,
    nb_attempts: int,
    temperature: float,
    top_p: float,
    model: str,
    use_chat_prompt: bool,
    stopwords: list[str],
    api_key: str | None,
    api_base_url: str | None,
    max_total_tokens: int,
    max_generated_tokens: int,
    prompt_context: PromptContext,
    console=console,
) -> tuple[list[str | None], int, int]:
    # prepare prompt context
    lean_declaration = node["lean_declarations"][0]
    original_file_content: str = lean_files[lean_declaration["file"]]
    original_lean_context = original_file_content[: lean_declaration["theorem_info"]["proof_start_idx"]]

    def compress_lean_context(lean_context: str, level: int = 0) -> str:
        if level == -1:
            return ""
        # if level >= 0:
        #     lean_context = LeanFilesProcessor(lean_declarations).remove_proofs(lean_declaration["file"], lean_context)
        #     # lean_context = LeanFilesProcessor(lean_declarations).remove_theorems(lean_declaration["file"], lean_context)
        if level >= 2:
            # remove lines in the middle of the context
            lines = lean_context.split("\n")
            keep_lines = len(lines) // (2**level)
            if keep_lines == 0:
                return ""
            lines = lines[:keep_lines] + ["... [TRUNCATED] ..."] + lines[-keep_lines:]
            lean_context = "\n".join(lines)

        return lean_context.strip()

    os.makedirs(output_folder, exist_ok=True)

    max_input_tokens = int(0.8 * (max_total_tokens - max_generated_tokens))

    with optional_status(f"Generating proofs for node {node['label']}...", enabled=verbose):
        if use_chat_prompt:
            match prompt_context:
                case PromptContext.NO_CONTEXT:
                    raise NotImplementedError("No context is not yet implemented for chat prompts.")
                case PromptContext.FILE_CONTEXT_NO_LEMMAS:

                    def instantiate_prompt(lean_context: str, level: int = 0) -> str:
                        # Get all lemma names from the blueprint graph
                        blueprint_lemmas = set()
                        for blueprint_node in blueprint_graph.nodes():
                            node_data = blueprint_graph.nodes[blueprint_node]
                            if "lean_declarations" in node_data:
                                for lean_decl in node_data["lean_declarations"]:
                                    if "full_name" in lean_decl:
                                        blueprint_lemmas.add(lean_decl["full_name"])

                        # Process the context by removing proofs and non-blueprint lemmas
                        lean_processor = LeanFilesProcessor(lean_declarations)
                        lean_context = lean_processor.remove_theorems(
                            lean_declaration["file"], original_lean_context, whitelist=blueprint_lemmas
                        )

                        # Additional compression if needed
                        lean_context = compress_lean_context(lean_context, level=level)

                        # Format the prompt
                        prompt = "Here is the Lean 4 context:\n```lean4\n" + lean_context.strip() + "\n```"
                        prompt += "\n\nUsing this context, formalize the following proof into Lean 4.\n"
                        prompt += f"```lean4\n{_node_informal_comment(node)}```"
                        prompt += "\nStart your formalization like this:\n```lean4\n:= by"
                        return prompt

                    # we assume files are less than 2^16=65536 lines of code, beyond that it is probably useless to try
                    messages = None
                    compress_level = 0
                    for compress_level in range(16):
                        messages = clean_messages(
                            [
                                {
                                    "role": "user",
                                    "content": instantiate_prompt(original_lean_context, level=compress_level),
                                }
                            ]
                        )
                        if token_counter(model=model, messages=messages) <= max_input_tokens:
                            break
                        else:
                            messages = None
                    if not messages:
                        compress_level = -1
                        messages = clean_messages(
                            [
                                {
                                    "role": "user",
                                    "content": instantiate_prompt(original_lean_context, level=compress_level),
                                }
                            ]
                        )
                        if token_counter(model=model, messages=messages) > max_input_tokens:
                            logger.warning(
                                f"Natural language context too large for node {node['label']}. Skipping formalization."
                            )
                            return [None for _ in range(nb_attempts)], 0, 0

                case PromptContext.FILE_CONTEXT:

                    def instantiate_prompt(lean_context: str, level: int = 0) -> str:
                        lean_context = compress_lean_context(lean_context, level=level)
                        prompt = "Here is the Lean 4 context:\n```lean4\n" + lean_context.strip() + "\n```"
                        prompt += "\n\nUsing this context, formalize the following proof into Lean 4.\n"
                        prompt += f"```lean4\n{_node_informal_comment(node)}```"
                        prompt += "\nStart your formalization like this:\n```lean4\n:= by"
                        return prompt

                    # we assume files are less than 2^16=65536 lines of code, beyond that it is probably useless to try
                    messages = None
                    compress_level = 0
                    for compress_level in range(16):
                        messages = clean_messages(
                            [
                                {
                                    "role": "user",
                                    "content": instantiate_prompt(original_lean_context, level=compress_level),
                                }
                            ]
                        )
                        if token_counter(model=model, messages=messages) <= max_input_tokens:
                            break
                        else:
                            messages = None
                    if not messages:
                        compress_level = -1
                        messages = clean_messages(
                            [
                                {
                                    "role": "user",
                                    "content": instantiate_prompt(original_lean_context, level=compress_level),
                                }
                            ]
                        )
                        if token_counter(model=model, messages=messages) > max_input_tokens:
                            logger.warning(
                                f"Natural language context too large for node {node['label']}. Skipping formalization."
                            )
                            return [None for _ in range(nb_attempts)], 0, 0

            # dump the messages to a file
            with open(
                os.path.join(output_folder, f"input_messages_compress_{compress_level}.json"), "w"
            ) as inputs_file:
                inputs_file.write(json.dumps(messages, indent=4, ensure_ascii=False))

            try:
                completion_response = completion(
                    api_key=api_key,
                    api_base=api_base_url,
                    model=model,
                    messages=messages,
                    max_tokens=max_generated_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=nb_attempts,
                    caching=True,
                    stop=stopwords,
                )
            except ContextWindowExceededError:
                logger.warning(f"Context window exceeded for node {node['label']}")
                return [None for _ in range(nb_attempts)], 0, 0

            predictions = [
                choice.message.content if choice.message.content and choice.message.content.strip() else None  # type: ignore
                for choice in completion_response.choices  # type: ignore
            ]
            predictions = [
                "\n\n".join(extract_lean_codes(prediction)) if prediction else None for prediction in predictions
            ]

        else:
            match prompt_context:
                case PromptContext.NO_CONTEXT:
                    compress_level = 0
                    prompt = f"Natural language version:\n{_node_informal_text(node)}\nTranslate the natural language version to a Lean 4 version:"
                    raise NotImplementedError("No context is not yet implemented for chat prompts.")
                case PromptContext.FILE_CONTEXT_NO_LEMMAS:

                    def instantiate_prompt(lean_context: str, level: int = 0) -> str:
                        # Get all lemma names from the blueprint graph
                        blueprint_lemmas = set()
                        for blueprint_node in blueprint_graph.nodes():
                            node_data = blueprint_graph.nodes[blueprint_node]
                            if "lean_declarations" in node_data:
                                for lean_decl in node_data["lean_declarations"]:
                                    if "full_name" in lean_decl:
                                        blueprint_lemmas.add(lean_decl["full_name"])

                        # Process the context by removing proofs and non-blueprint lemmas
                        lean_processor = LeanFilesProcessor(lean_declarations)
                        lean_context = lean_processor.remove_theorems(
                            lean_declaration["file"], original_lean_context, whitelist=blueprint_lemmas
                        )

                        # Additional compression if needed
                        lean_context = compress_lean_context(lean_context, level=level)

                        # Format the prompt
                        prompt = (
                            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n"
                            + lean_context.strip()
                        ).strip()
                        prompt += "\n\nUsing this context, formalize the following proof into Lean 4.\n"
                        prompt += f"```lean4\n{_node_informal_comment(node)}```"
                        prompt += "\nStart your formalization like this:\n```lean4\n:= by"
                        return prompt

                    # we assume files are less than 2^16=65536 lines of code, beyond that it is probably useless to try
                    prompt = None
                    compress_level = 0
                    for compress_level in range(16):
                        prompt = instantiate_prompt(original_lean_context, level=compress_level)
                        if token_counter(model=model, text=prompt) <= max_input_tokens:
                            break
                        else:
                            prompt = None
                    if not prompt:
                        compress_level = -1
                        prompt = instantiate_prompt(original_lean_context, level=compress_level)
                        if token_counter(model=model, text=prompt) > max_input_tokens:
                            logger.warning(
                                f"Natural language context too large for node {node['label']}. Skipping formalization."
                            )
                            return [None for _ in range(nb_attempts)], 0, 0

                case PromptContext.FILE_CONTEXT:

                    def instantiate_prompt(lean_context: str, level: int = 0) -> str:
                        lean_context = compress_lean_context(lean_context, level=level)
                        lean_context += " := by\n" + _node_informal_comment(node)
                        return (
                            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n"
                            + lean_context.strip()
                        ).strip()

                    # we assume files are less than 2^16=65536 lines of code, beyond that it is probably useless to try to compress it
                    prompt = None
                    compress_level = 0
                    for compress_level in range(16):
                        prompt = instantiate_prompt(original_lean_context, level=compress_level)
                        if token_counter(model=model, text=prompt) <= max_input_tokens:
                            break
                        else:
                            prompt = None
                    if not prompt:
                        compress_level = -1
                        prompt = instantiate_prompt(original_lean_context, level=compress_level)
                        if token_counter(model=model, text=prompt) > max_input_tokens:
                            logger.warning(
                                f"Natural language context too large for node {node['label']}. Skipping formalization."
                            )
                            return [None for _ in range(nb_attempts)], 0, 0

            # dump the context to a file
            with open(os.path.join(output_folder, f"input_context_compress_{compress_level}.txt"), "w") as context_file:
                context_file.write(prompt)

            try:
                completion_response = text_completion(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_generated_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=nb_attempts,
                    api_key=api_key,
                    api_base=api_base_url,
                    stop=stopwords,
                    caching=True,
                )
            except ContextWindowExceededError:
                logger.warning(f"Context window exceeded for node {node['label']}")
                return [None for _ in range(nb_attempts)], 0, 0

            predictions = [
                choice.text if choice.text and choice.text.strip() else None  # type: ignore
                for choice in completion_response.choices  # type: ignore
            ]

        # dump the completion to a file
        with open(os.path.join(output_folder, "raw_completion.json"), "w") as completion_file:
            completion_file.write(completion_response.model_dump_json(indent=4))  # type: ignore

    input_tokens = completion_response.usage.prompt_tokens  # type: ignore
    output_tokens = completion_response.usage.completion_tokens  # type: ignore

    if verbose:
        console.print(f"Input tokens: {input_tokens}")
        console.print(f"Output tokens: {output_tokens}")

    return predictions, input_tokens, output_tokens


@dataclass
class ProcessPredictionsInput:
    predictions: list[str | None]
    lean_declaration: dict
    original_file_content: str
    node: dict
    output_folder: str
    repl_config: LeanREPLConfig
    timeout_context: int
    timeout_per_prediction: int


@dataclass
class PredictionEvaluationResult:
    lean_proof: str | None
    well_typed: bool = False
    error: str | None = None


def _process_predictions(args: ProcessPredictionsInput) -> tuple[str, str, list[PredictionEvaluationResult]]:
    decl_proof_gt = args.lean_declaration["decl"]
    decl = args.lean_declaration["theorem_info"]["declsig"] + " := by"

    lean_server = AutoLeanServer(args.repl_config)

    # Prepare the Lean context
    original_lean_context = args.original_file_content[: args.lean_declaration["start_idx"]]

    # check if the last line ends with " in" and remove " in" if it does
    trimmed_context = trim_comments_end(original_lean_context).rstrip()
    if "\n" in trimmed_context:
        lean_context, last_line = trimmed_context.rsplit("\n", 1)
    else:
        lean_context = ""
        last_line = trimmed_context
    if last_line.endswith(" in"):
        lean_context += "\n" + last_line[:-3]
    else:
        lean_context = original_lean_context

    # Ensure lean_context is not empty (Command requires at least 1 character)
    if not lean_context.strip():
        lean_context = "-- context stub"

    # Load the Lean context
    try:
        lean_context_output = lean_server.run(
            Command(cmd=lean_context), add_to_session_cache=True, timeout=args.timeout_context
        )
        # check if the context is valid
        if isinstance(lean_context_output, LeanError) or not lean_context_output.lean_code_is_valid():
            print_lean_context = lean_context
            if len(lean_context) > 1000:
                print_lean_context = lean_context[:500] + "\n\n... [TRUNCATED] ...\n\n" + lean_context[-500:]
            raise Exception("Invalid Lean context:\n" + str(lean_context_output) + "\n" + print_lean_context)
        context_env = lean_context_output.env
    except (TimeoutError, EOFError, json.JSONDecodeError) as e:
        print_lean_context = lean_context
        if len(lean_context) > 1000:
            print_lean_context = lean_context[:500] + "\n\n... [TRUNCATED] ...\n\n" + lean_context[-500:]
        raise Exception("Error while running the Lean context. Lean file:\n" + print_lean_context) from e

    # before doing anything, we check if we have at least one non-empty prediction
    if not any(args.predictions):
        return (
            args.node["label"],
            decl,
            [PredictionEvaluationResult(lean_proof=None) for _ in args.predictions],
        )

    # deduplicate the predictions to avoid running the same code multiple times
    dedup_predictions = Counter(args.predictions)

    with jsonlines.open(
        os.path.join(args.output_folder, "postprocessed_dedup_predictions.jsonl"), "w"
    ) as prediction_file:
        prediction_file.write_all({"prediction": prediction} for prediction in dedup_predictions)

    # dump the Lean codes with the context to a file
    for i, lean_proof in enumerate(dedup_predictions):
        if lean_proof:
            with open(os.path.join(args.output_folder, f"attempt_{i}.lean"), "w") as lean_file:
                lean_file.write(lean_context + "\n" + decl + "\n" + lean_proof)
    with open(os.path.join(args.output_folder, "ground_truth.lean"), "w") as lean_file:
        lean_file.write(lean_context + "\n" + decl_proof_gt)

    # check that the ground truth is well-typed. It should be, otherwise it means we have a problem with the context
    ground_truth_output = lean_server.run(
        Command(cmd=decl_proof_gt, env=context_env), timeout=args.timeout_per_prediction
    )
    if isinstance(ground_truth_output, LeanError) or not ground_truth_output.lean_code_is_valid():
        raise Exception(f"Invalid ground truth Lean code:\n{str(ground_truth_output)}\n{decl_proof_gt}")

    tmp_res: list[PredictionEvaluationResult] = []
    for i, lean_proof in enumerate(dedup_predictions):
        tmp_res.append(PredictionEvaluationResult(lean_proof=lean_proof))

        if not lean_proof:
            continue

        try:
            lean_output = lean_server.run(
                Command(cmd=decl + "\n" + lean_proof, env=context_env), timeout=args.timeout_per_prediction
            )
            if isinstance(lean_output, LeanError):
                continue
            tmp_res[-1].well_typed = lean_output.lean_code_is_valid(allow_sorry=False)

            # dump the Lean server output
            with open(os.path.join(args.output_folder, f"type_check_output_{i}.json"), "w") as lean_output_file:
                lean_output_file.write(json.dumps(lean_output.model_dump(mode="json"), indent=4, ensure_ascii=False))

        except ValueError as e:
            tmp_res[-1].error = str(e)
        except Exception as e:
            lean_server.restart()
            tmp_res[-1].error = str(e)

    lean_code_to_result = {result.lean_proof: result for result in tmp_res}
    return (
        args.node["label"],
        decl_proof_gt,
        [lean_code_to_result[prediction] for prediction in args.predictions],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate autoformalization")
    parser.add_argument("--benchmark-config", type=str, required=True, help="Path to benchmark YAML config")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model YAML config")
    args = parser.parse_args()

    with open(args.benchmark_config, "r") as f:
        benchmark_config = yaml.safe_load(f)
    projects = benchmark_config.get("repositories", [])

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    # Extract model parameters from model config, with defaults as fallback
    model_name = model_config.get("model", "gpt-4o-2024-05-13")
    temperature = model_config.get("temperature", 0.0)
    top_p = model_config.get("top_p", 0.95)
    nb_samples = model_config.get("nb_samples", 1)
    max_total_tokens = model_config.get("max_total_tokens", 4096)
    max_generated_tokens = model_config.get("max_generated_tokens", 512)
    use_chat_prompt = model_config.get("use_chat_prompt", True)
    stopwords = model_config.get("stopwords", ["```\n", ":= by", "sorry"])
    n_processes = model_config.get("n_processes", 15)
    gen_processes = model_config.get("gen_processes", None)
    prompt_context = PromptContext[model_config.get("prompt_context", "FILE_CONTEXT")]
    api_key = model_config.get("api_key", None)
    api_base_url = model_config.get("api_base_url", None)

    verbose = True

    traced_repos_dir = os.path.join(ROOT_DIR, "traced_repos")

    for repo in projects:
        # Construct project directory and project name using benchmark info
        git_url = repo["git_url"]
        commit = repo["commit"]
        project_name_bench = repo["project_name"]
        project_dir = f"{git_url.split('/')[-1]}_{commit}"
        project_root_dir = os.path.join(traced_repos_dir, project_dir)
        lean_project_root_dir = os.path.join(project_root_dir, git_url.split("/")[-1])

        console.rule(f"Formalizing {project_name_bench}")

        with jsonlines.open(os.path.join(project_root_dir, "blueprint_to_lean.jsonl")) as reader:
            blueprint_to_lean = list(reader)
        with jsonlines.open(os.path.join(project_root_dir, "lean_files.jsonl")) as reader:
            lean_files = list(reader)
            assert len({file["file"] for file in lean_files}) == len(lean_files), "Duplicate Lean files found"
            lean_files = {file["file"]: file["content"] for file in lean_files}
        with jsonlines.open(os.path.join(project_root_dir, "lean_declarations.jsonl")) as reader:
            lean_declarations = list(reader)

        output_folder = os.path.join(
            ROOT_DIR,
            "results",
            "proof",
            project_name_bench,
            model_name.split("/")[-1],
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        # Copy the benchmark and model config files to the result folder
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "benchmark_config.yaml"), "w") as f:
            yaml.safe_dump(benchmark_config, f)
        with open(os.path.join(output_folder, "model_config.yaml"), "w") as f:
            yaml.safe_dump(model_config, f)

        agent = ProvingEvaluation(
            blueprint_with_lean=blueprint_to_lean,
            lean_files=lean_files,
            lean_declarations=lean_declarations,
            project_dir=lean_project_root_dir,
        )

        total_input_tokens, total_output_tokens = agent.run(
            output_folder=output_folder,
            nb_attempts=nb_samples,
            top_p=top_p,
            max_total_tokens=max_total_tokens,
            max_generated_tokens=max_generated_tokens,
            verbose=verbose,
            model=model_name,
            temperature=temperature,
            stopwords=stopwords,
            api_key=api_key,
            api_base_url=api_base_url,
            use_chat_prompt=use_chat_prompt,
            n_processes=n_processes,
            prompt_context=prompt_context,
            gen_processes=gen_processes,
        )
