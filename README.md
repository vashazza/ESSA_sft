# ESSA: Evolved Safety Specification Alignment

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b)](https://openreview.net/forum?id=Imcv0X7c36)
[![PDF](https://img.shields.io/badge/PDF-Download-blue)](https://openreview.net/pdf?id=Imcv0X7c36)
[![Venue](https://img.shields.io/badge/AI4GOOD-ICML%202026-2c5282)](https://openreview.net/forum?id=Imcv0X7c36)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Code and data for **ESSA: Evolved Safety Specification Alignment** (AI4GOOD Workshop, ICML 2026).
Dongcho Park, Yeowon Jung, Sung Jun Cheon.

## Overview

Safety alignment is often miscalibrated: models over-refuse benign requests yet stay vulnerable to multi-turn jailbreaks. ESSA is a post-training framework that teaches a model to *deliberate* with explicit safety specifications conditioned on both the **domain** and **task** of a request. Specifications are evolved per `(domain, task)` pair from a three-tier constitution via an LLM-driven evolutionary loop; a corpus of deliberation traces is then synthesized under them to fine-tune a target model.

This repository covers the **synthesis and SFT stage**: it generates spec-grounded traces with a teacher model, formats them for Qwen3 thinking mode, fine-tunes the model, and evaluates it. The evolved specifications are an input (see [Specifications](#specifications)).

## Repository structure

```
code/                  # SFT training, inference, GPT-based evaluation
config/                # training_config.yaml (model, data, training, eval)
data/code/             # Teacher-model trace generation (runner, spec loader, templates)
data/SFT_seed/         # Seed prompts and distribution statistics
data/sft_data/         # Final traces (evolved domain-and-task specs)
data/general_only_data/# Ablation: domain/task-agnostic spec
data/no_spec_data/     # Ablation: no spec
```

## Data

The seed set has 8,000 prompts: 5,000 utility (Nemotron-Post-Training-Dataset-V2) and 3,000 safety (Aegis-AI-Content-Safety-2.0 ~900, FalseReject ~1,500, JailbreakV-28k ~600).

Synthesized files are named `{Domain}_{Source}.json`, with domain in `F/G/H/L/T` (Finance, General, Healthcare, Legal, Telecom) and source in `AG/FR/JB` (Aegis, FalseReject, JailbreakV). Each record holds the prompt and a `teacher_response` split into a `##thought` (deliberation) and `##response` (answer), reformatted by `data_preprocessing.py` into the `<think>...</think>` chat format.

## Setup

```bash
git clone https://github.com/vashazza/ESSA_sft.git
cd ESSA_sft
pip install -r requirements.txt
```

A CUDA GPU is required for training and trace generation; evaluation needs `OPENAI_API_KEY`.

## Usage

```bash
# 1. Synthesize traces under the evolved specs
python data/code/teacher_runner.py --csv data/SFT_seed/aegis_900.csv \
  --out data/sft_data/G_AG_OUT.json --spec_root /path/to/SPEC \
  --system_tpl data/code/teacher_system.txt --user_tpl data/code/teacher_user.txt

# 2. Preprocess into train/val splits
python code/data_preprocessing.py --mode pipeline \
  --sft_dir data/sft_data --output_dir processed_data --val_ratio 0.1

# 3. Fine-tune (LoRA, settings in the YAML)
python code/train_qwen_thinking.py --config config/training_config.yaml

# 4. Generate base vs. fine-tuned responses
python code/inference.py --config config/training_config.yaml \
  --base_model Qwen/Qwen3-4B --finetuned_model ./qwen3_finetuned_<timestamp>

# 5. Pairwise evaluation (GPT-4o)
python code/evaluate_responses_only.py --responses_file ./responses.json \
  --config config/training_config.yaml
```

Target model: Qwen3-4B in thinking mode, LoRA (rank 16, α 32), seq length 4,096, 2 epochs, lr 1e-5 (cosine), effective batch 16, bf16. See `config/training_config.yaml` for the full configuration.

## Specifications

Evolved specs are consumed as JSON under a `SPEC/` root organized as `SPEC/{Domain}/{Task}.json` (a list of rules, or a `{"specifications": [...]}` / `{"rules": [...]}` object). Pass this path via `--spec_root` (see `data/code/teacher_spec.py`).

## Ethics

Some prompts are adversarial by design. The release supports reproducible safety research and is not intended to elicit harmful behavior. Synthesized traces are teacher-generated research artifacts and may contain errors. Users must comply with the licenses of all upstream models and datasets.

## License

Code is released under the MIT License (`LICENSE`); the paper is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Seed prompts derive from third-party datasets and remain under their original terms.

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
