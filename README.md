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

### End-to-end workflow (inference → ranked spacers → picking → primers)

Run steps **in order** from the repo root with `PYTHONPATH=src` (or your usual `export`).

1. **Inference (scan + extract + score)** — `spacerModel.inference.genomic_spacer_predict_cnn` walks each FASTA record for **YN-PAM** sites, **extracts** 32 bp spacers on both strands, and scores them with the CNN. It writes **`ranked_sequences.xlsx`** (all genes) and **`ranked_sequences_<gene>.xlsx`** per gene in the **directory containing** your `--output` argument (see CLI below). Those tables are the **ranked spacer** inputs for the next steps.

2. **Picking (curated sets + Primer-BLAST)** — `spacerModel.inference.picking` reads a ranked table (typically `ranked_sequences.xlsx` or a per-gene workbook) plus the **same reference FASTA** used in step 1. For each gene it can take the **best**-ranked spacers, optionally **worst**-ranked, and **random** picks from the rest, and runs **NCBI Primer-BLAST** (per spacer) to attach primer pairs. Example:

   ```bash
   PYTHONPATH=src python -m spacerModel.inference.picking \
     --input output/inference/ranked_sequences.xlsx \
     --fasta path/to/same_genome.fa \
     --output output/inference/picked_spacers.xlsx
   ```

   Use `--number-top-picked`, `--number-bottom-picked`, `--number-random-picked`, and `--gene` (single gene or `all`) as needed; see `--help`.

3. **Primer search only (optional)** — If you want Primer-BLAST on **every row** of a ranked table **without** the picking logic, use `spacerModel.inference.primer` (see below). Picking already performs primer search for the rows it selects; use `primer.py` when you need a fully annotated ranked sheet first.

**What you need**

- A checkpoint file (`.pt`) with `model_kind == "cnn"`, trained on **flat one-hot** DNA (`--embedding onehot`), consistent with how the model was saved.
- A **FASTA** file (e.g. `.fa`). Each ``>`` header line is stored **verbatim** (text after ``>``, stripped) in the output **Gene** column. For **picking** (`--gene`) and **per-gene density plots**, the matchable “symbol” is the substring **before the first underscore** when the header contains `_` (e.g. ``>MYO7A_chr7_...`` → ``MYO7A``). If there is **no** underscore, the whole header is used. Underscores are **not** required; they only help separate a short gene tag from extra text in the header. Plain text **without** any ``>`` line is treated as a single sequence with ``Gene`` set to ``(no_header)``.

**Convention**

- Spacer length is **32 bp** by default; it must match the checkpoint (`n_features = 4 × length`).
- Sites are scanned on **both strands** for a **YN** PAM (first base Y ∈ {C, T}) immediately 5′ of the spacer: `[PAM(2)][SPACER(32)]` on the targeting strand. Output coordinates are **1-based** on the forward reference.
- **Flanking sequence (default 400 bp each end):** Each FASTA record should extend **at least ~400 nt upstream and ~400 nt downstream** beyond the genomic interval where you want protospacers scored. By default the scanner **skips** the outer **400 bp** on each end of the record (edge reserve) so hits stay away from margins while **coordinates stay 1-based on the full record** (for primer tools). If your sequence is shorter than ``2 × edge_reserve``, that record is skipped. For a **full-length** scan of the record, use ``--no-omit-edge-bases`` (or ``--spacer-search-edge-reserve 0``). See ``--help``.

**CLI**

From the repository root (with `PYTHONPATH=src`):

```bash
python -m spacerModel.inference.genomic_spacer_predict_cnn \
  --checkpoint path/to/CNN_best_model.pt \
  --fasta path/to/genome.fa \
  --output output/inference/ranked_sequences.xlsx
```

Per-gene workbooks and `ranked_sequences.xlsx` are written in the **parent directory** of `--output` (here `output/inference/`). Inference runs on CPU. See ``--help`` on that module for options (e.g. edge reserve for padded FASTA, plots).

## Primer-BLAST (optional, full-table annotation)

For **full-table** Primer-BLAST annotation (not the curated pick workflow above), ``src/spacerModel/inference/primer.py`` calls [NCBI Primer-BLAST](https://www.ncbi.nlm.nih.gov/tools/primer-blast/) on ranked spacer tables; use ``--help`` for inputs and flags.

```bash
PYTHONPATH=src python -m spacerModel.inference.primer \
  --input path/to/ranked_sequences.xlsx \
  --fasta path/to/same_genome.fa \
  --output path/to/ranked_with_primers.xlsx
```

## Other functionality

- **Dataset analysis and logos**: `src/spacerModel/data/analyze_dataset.py`
- **Simple hypothesis tests on data**:
`src/spacerModel/simple_modeling/simple_hypothesis_test.py`
- **Training / hyperparameter search**: see `src/spacerModel/training/` (`train_torch.py`, `torch_hyperopt.py`, `retrain_optimized_models.py`, etc.).
- **CNN explainability** (saliency, layer activations): `src/spacerModel/interpretation/explain_cnn_layers.py`.
