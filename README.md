# RLM: Evaluation of Autoformalization Methods on Research-Level Mathematics

This repository features the code for the evaluation of autoformalization methods on the RLM25 benchmark.

## Project setup

> [!IMPORTANT]
> Set the environment variables `GITHUB_ACCESS_TOKEN` (see [GitHub documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#personal-access-tokens-classic)) and `OPENAI_API_KEY`.


### Development inside Docker (recommended)

In VS Code, run the **Dev Containers: Open Folder in Container...** command from the Command Palette (F1). The `.devcontainer` folder contains the necessary configuration and will take care of setting up the environment.

### Local installation

Requirements:

- Python >= 3.10
- git
- [Lean 4](https://leanprover-community.github.io/get_started.html)

Install Python project:

    pip install -e .

## RLM25

Prepare the RLM25 dataset:

    python scripts/extract_benchmark.py  --config configs/benchmark/rlm25.yaml

Run statement autoformalization evaluation:

    python scripts/eval_statement_autoformalization.py --benchmark-config configs/benchmark/rlm25.yaml --model-config configs/models/gpt-4o_greedy.yaml

Run proof autoformalization evaluation:

    python scripts/eval_proof_autoformalization.py --benchmark-config configs/benchmark/rlm25.yaml --model-config configs/models_proof/gpt-4o_greedy.yaml

## Citation

[Improving Autoformalization using Type Checking](https://arxiv.org/abs/2406.07222)

```bibtex
@misc{poiroux2024improvingautoformalizationusingtype,
    title={Improving Autoformalization using Type Checking}, 
    author={Auguste Poiroux and Gail Weiss and Viktor Kunčak and Antoine Bosselut},
    year={2024},
    eprint={2406.07222},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2406.07222}, 
}
```
