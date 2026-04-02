# evoCAST spacer model

Repository for training and evaluating models that predict functional enrichment (log2-FE) from DNA spacer sequences, with utilities for analysis and **genomic scanning** using a trained CNN.

## Clone the repository

```bash
git clone https://github.com/TereAuge2/evoCAST_spacerModel.git
cd <repo-name>
```

Use SSH instead if you prefer: `git clone git@github.com:TereAuge2/evoCAST_spacerModel.git`

## Setup

- **Python** 3.10+ recommended (see `requirements.txt` comments).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Run modules with the package on the path:

  ```bash
  export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
  ```

  Or install the package in editable mode if you add a `pyproject.toml` / `setup.cfg`; until then, `PYTHONPATH=src` is the usual approach.

- **PyTorch**: install a build that matches your machine (CPU or CUDA). The `torch>=2.0` line in `requirements.txt` may need to be satisfied via [pytorch.org](https://pytorch.org) instructions for GPU wheels.

## Making predictions (CNN, genomic FASTA)

This flow scores **candidate protospacers** in a genomic sequence using a saved **CNN** checkpoint produced by the training pipeline (`save_torch_best_checkpoint` / `final_models/CNN_best_model.pt`).

**What you need**

- A checkpoint file (`.pt`) with `model_kind == "cnn"`, trained on **flat one-hot** DNA (`--embedding onehot`), consistent with how the model was saved.
- A **FASTA** or plain-text file. For FASTA, each ``>`` header line defines the **gene** label for that sequence (full header text after ``>`` is stored in a **Gene** column). Multiple records are concatenated in the output table. Plain text **without** any ``>`` line is treated as a single sequence with ``Gene`` set to ``(no_header)``.

**Convention**

- Spacer length is **32 bp** by default; it must match the checkpoint (`n_features = 4 × length`).
- Sites are scanned on **both strands** for a **YN** PAM (first base Y ∈ {C, T}) immediately 5′ of the spacer: `[PAM(2)][SPACER(32)]` on the targeting strand. Output coordinates are **1-based** on the forward reference.
- Optional **edge exclusion** for the scan (default 400 bp per end); use ``--no-omit-edge-bases`` for a full-length scan. See ``--help``.

**CLI**

From the repository root (with `PYTHONPATH=src`):

```bash
python -m spacerModel.inference.genomic_spacer_predict_cnn \
  --checkpoint path/to/CNN_best_model.pt \
  --fasta path/to/genome.fa \
  --output path/to/ranked_sites.xlsx
```

Inference runs on CPU. See ``--help`` on that module for options (e.g. edge reserve for padded FASTA, plots).

## Primer-BLAST (optional)

``src/spacerModel/inference/primer.py`` calls [NCBI Primer-BLAST](https://www.ncbi.nlm.nih.gov/tools/primer-blast/) to annotate ranked spacer tables; use ``--help`` for inputs and flags.

```bash
PYTHONPATH=src python -m spacerModel.inference.primer \
  --input path/to/ranked_sequences_*.xlsx \
  --fasta path/to/same_genome.fa \
  --output path/to/ranked_with_primers.xlsx
```

## Other functionality

- **Dataset analysis and logos**: `src/spacerModel/data/analyze_dataset.py`
- **Simple hypothesis tests on data**:
`src/spacerModel/simple_modeling/simple_hypothesis_test.py`
- **Training / hyperparameter search**: see `src/spacerModel/training/` (`train_torch.py`, `torch_hyperopt.py`, `retrain_optimized_models.py`, etc.).
- **CNN explainability** (saliency, layer activations): `src/spacerModel/interpretation/explain_cnn_layers.py`.
