# ESSA: Evolved Safety Specification Alignment

This repository contains the data-synthesis and supervised fine-tuning (SFT) code for **ESSA: Evolved Safety Specification Alignment**, accepted at the *Trustworthy AI for Good (AI4GOOD) Workshop, ICML 2026*.

> **Paper:** [OpenReview](https://openreview.net/forum?id=Imcv0X7c36) · [PDF](https://openreview.net/pdf?id=Imcv0X7c36)
> **Authors:** Dongcho Park, Yeowon Jung, Sung Jun Cheon

## Overview

Safety alignment in large language models is often miscalibrated: models over-refuse benign requests while remaining vulnerable to multi-turn jailbreaks that gradually poison the conversational context. ESSA is a post-training framework that teaches a model to *deliberate* with explicit safety specifications that are conditioned on both the **domain** and the **task** of a request, rather than relying on a single monolithic policy.

The full method has three components. A three-tier hierarchical constitution sets global priorities and interpretation rules. From this constitution, compact safety-specification sets are automatically evolved for each `(domain, task)` pair through an LLM-driven evolutionary loop with multi-perspective judging and redundancy control. A training corpus of deliberation traces and responses is then synthesized under the evolved specifications and used to fine-tune a target model.

This repository implements the **synthesis and SFT stage**: given a set of evolved specifications, it generates spec-grounded deliberation traces with a teacher model, converts them into the Qwen3 thinking format, fine-tunes the target model, and evaluates the result. The evolved specifications themselves are treated as an input (see [Specifications](#specifications)).

## Repository structure

```
.
├── code/                          # Training, inference, and evaluation
│   ├── train_qwen_thinking.py     # SFT entry point (YAML-driven)
│   ├── data_preprocessing.py      # Trace parsing, formatting, train/val split
│   ├── thinking_data_collator.py  # Collator for the thinking (<think>) format
│   ├── model_utils.py             # Model/tokenizer/dataset helpers
│   ├── inference.py               # Base vs. fine-tuned response generation
│   └── evaluate_responses_only.py # GPT-based pairwise evaluation
├── config/
│   └── training_config.yaml       # All model, data, training, and eval settings
├── data/
│   ├── code/                      # Teacher-model trace generation
│   │   ├── teacher_runner.py      # Batch generation over a prompt CSV
│   │   ├── teacher_spec.py        # Loads specs and renders prompts
│   │   ├── teacher_system.txt     # System template (deliberation procedure)
│   │   └── teacher_user.txt       # User template
│   ├── SFT_seed/                  # Seed prompts and distribution statistics
│   ├── sft_data/                  # Final traces synthesized under evolved specs
│   ├── general_only_data/         # Ablation: domain/task-agnostic specification
│   └── no_spec_data/              # Ablation: no specification
├── requirements.txt
└── README.md
```

## Data

### Seed prompts

The seed set contains 8,000 prompts that balance general utility against safety-critical behavior:

| Subset  | Source                              | Size  | Purpose                              |
|---------|-------------------------------------|-------|--------------------------------------|
| Utility | Nemotron-Post-Training-Dataset-V2   | 5,000 | Conversational data with reasoning traces |
| Safety  | Aegis-AI-Content-Safety-2.0         | ~900  | Harmful-content refusal              |
| Safety  | FalseReject                         | ~1,500| Over-refusal of benign requests      |
| Safety  | JailbreakV-28k                      | ~600  | Adversarial / jailbreak prompts      |

Per-file domain and task distributions are reported in `data/SFT_seed/distribution`.

### Synthesized traces

Synthesized files follow the convention `{Domain}_{Source}.json`, where the domain code is one of `F` (Finance & Banking), `G` (General), `H` (Healthcare & Medicine), `L` (Legal & Regulatory), `T` (Telecommunications), and the source code is `AG` (Aegis), `FR` (FalseReject), or `JB` (JailbreakV). Three variants are provided to support the ablations in the paper:

- `sft_data/` — traces generated under the **evolved, domain-and-task-specific** specifications (the main configuration; files carry an `_OUT` suffix).
- `general_only_data/` — traces generated under a **single domain/task-agnostic** specification.
- `no_spec_data/` — traces generated with **no** specification.

Each record stores the original prompt and a `teacher_response` field containing a `##thought` section (the deliberation) and a `##response` section (the user-facing answer), which `data_preprocessing.py` separates and reformats into the Qwen3 `<think>...</think>` chat format.

## Installation

```bash
git clone https://github.com/vashazza/ESSA_sft.git
cd ESSA_sft
pip install -r requirements.txt
```

Training and trace generation expect a CUDA-capable GPU. `bitsandbytes` and `flash-attn` are optional and may be removed if unsupported on the target platform. GPT-based evaluation requires an `OPENAI_API_KEY` environment variable.

## Usage

The pipeline proceeds from specifications to evaluation. Paths below assume the working directory is the repository root; adjust them to your setup.

**1. Synthesize deliberation traces.** A teacher model (Qwen3-4B-Instruct-2507 by default) generates spec-grounded `##thought`/`##response` pairs for each prompt in a CSV, loading the specification for that row's `(domain, task)`:

```bash
python data/code/teacher_runner.py \
  --csv data/SFT_seed/aegis_900.csv \
  --out data/sft_data/G_AG_OUT.json \
  --system_tpl data/code/teacher_system.txt \
  --user_tpl data/code/teacher_user.txt \
  --spec_root /path/to/SPEC \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --max_new_tokens 4096
```

**2. Preprocess into the training format.** Parse the traces, convert them to the thinking chat format, and split into train/validation:

```bash
python code/data_preprocessing.py \
  --mode pipeline \
  --sft_dir data/sft_data \
  --output_dir processed_no_spec_data \
  --val_ratio 0.1
```

**3. Fine-tune the target model.** All hyperparameters are read from the YAML config:

```bash
python code/train_qwen_thinking.py --config config/training_config.yaml
```

**4. Generate responses** from the base and fine-tuned models for comparison:

```bash
python code/inference.py \
  --config config/training_config.yaml \
  --base_model Qwen/Qwen3-4B \
  --finetuned_model ./qwen3_finetuned_<timestamp>
```

**5. Evaluate** the generated responses with rubric-based pairwise judging:

```bash
python code/evaluate_responses_only.py \
  --responses_file ./path/to/responses.json \
  --config config/training_config.yaml
```

## Training configuration

The target model is **Qwen3-4B** fine-tuned in the thinking format (`enable_thinking: true`), with the deliberation tokens kept in the loss rather than masked. Fine-tuning uses LoRA (rank 16, α 32, dropout 0.05) on all attention and MLP projections. Sequences are packed to a maximum length of 4,096 tokens. The model is trained for 2 epochs with a learning rate of 1e-5 under a cosine schedule (3% warmup), an effective batch size of 16 (per-device batch 1 with 16 gradient-accumulation steps), bf16 precision, and seed 42. Evaluation uses GPT-4o for pairwise judgments along helpfulness, accuracy, clarity, and safety. All settings can be changed in `config/training_config.yaml`; runs are logged to Weights & Biases.

## Specifications

The evolved `(domain, task)` specification sets are produced by the ESSA evolutionary loop and consumed here as JSON files under a `SPEC/` root, organized as `SPEC/{Domain}/{Task}.json`. Each file is a list of rules, or a `{"specifications": [{"text": ...}]}` / `{"rules": [...]}` object (see `data/code/teacher_spec.py`). Point `--spec_root` at this directory when generating traces.

## Ethics statement

This project studies safety alignment, and some seed prompts are adversarial or describe harmful scenarios by design. The data and code are released to support reproducible safety research and the evaluation of refusal calibration and jailbreak robustness. They are not intended for, and should not be used to, elicit or operationalize harmful behavior. The synthesized traces are produced by a teacher model and may contain errors or residual biases; they should be treated as research artifacts rather than authoritative guidance. Downstream users are responsible for complying with the licenses and acceptable-use terms of all upstream models and datasets.

## License

The paper is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in this repository is released under the MIT License; see `LICENSE`. The seed prompts are derived from third-party datasets (Nemotron-Post-Training-Dataset-V2, Aegis-AI-Content-Safety-2.0, FalseReject, JailbreakV-28k) and remain subject to their respective licenses and terms of use.

## Acknowledgements

This work builds on the Qwen3 model family and on the Nemotron-Post-Training-Dataset-V2, Aegis-AI-Content-Safety-Dataset-2.0, FalseReject, and JailbreakV-28k datasets. We thank their authors and maintainers.

## Citation

```bibtex
@inproceedings{park2026essa,
  title     = {{ESSA}: Evolved Safety Specification Alignment},
  author    = {Park, Dongcho and Jung, Yeowon and Cheon, Sung Jun},
  booktitle = {Trustworthy AI for Good (AI4GOOD) Workshop, ICML},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Imcv0X7c36}
}
```
