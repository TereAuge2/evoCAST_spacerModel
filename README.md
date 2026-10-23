# evoCAST spacer model

Repository for training and evaluating models that predict functional enrichment (log2-FE) from DNA spacer sequences, with utilities for analysis and **genomic scanning** using a trained CNN.

**End-to-end capabilities**

1. Scan a genomic FASTA for YN-PAM protospacer sites (both strands).
2. Score candidate 32 bp spacers with a trained **CNN** (`models/CNN_best_model.pt`).
3. Rank spacers per gene and plot predicted log2-FE density.
4. Pick top spacers and run **NCBI Primer-BLAST** to suggest amplicon primers.
5. Build **in-silico donor plasmids** — insert the top picked spacer (or top model-ranked spacer if primers are skipped) into the PseCAST donor template and export annotated GenBank files.

## Quick start (Google Colab)

The easiest way to run the full pipeline without a local install is the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TereAuge2/evoCAST_spacerModel/blob/main/notebooks/evoCAST_spacer_Colab_v2.ipynb)

**Notebook:** [`notebooks/evoCAST_spacer_Colab_v2.ipynb`](notebooks/evoCAST_spacer_Colab_v2.ipynb) (older version: [`notebooks/evoCAST_spacer_Colab.ipynb`](notebooks/evoCAST_spacer_Colab.ipynb))

**What to do**

1. Open the notebook in Colab (link above).
2. Set **User settings** (optional GitHub token for a private clone; number of spacers to send through Primer-BLAST).
3. Paste your **FASTA** in the sequence cell — include **400 nt upstream and 400 nt downstream** of the region you want to scan.
4. **Runtime → Run all**.

**Outputs** (zipped from the notebook): ranked spacer tables (Excel), density plots, primer-picked spacers (`Top_spacers_with_primers.xlsx`), and donor plasmids under `donor_plasmids/` as `PseCAST_{gene}_rank{n}_donor.gbk` (one per gene; first primer pick; feature labeled `spacer`).

The notebook clones this repo, installs dependencies from [`notebooks/requirements-colab.txt`](notebooks/requirements-colab.txt), and uses the bundled checkpoint. Primer-BLAST requires network access and can take ~1 minute per spacer.

## Clone the repository

```bash
git clone https://github.com/TereAuge2/evoCAST_spacerModel.git
cd evoCAST_spacerModel
```

Use SSH instead if you prefer: `git clone git@github.com:TereAuge2/evoCAST_spacerModel.git`

## Local setup

Use this path if you want the CLI, training scripts, or to run the notebook locally (e.g. Jupyter).

- **Python** 3.10+ recommended.
- **Clone** the repo (see above) and install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  For a minimal inference-only environment, Colab-style deps are listed in [`notebooks/requirements-colab.txt`](notebooks/requirements-colab.txt) (you still need PyTorch and the checkpoint).

- **PyTorch**: install a build that matches your machine (CPU or CUDA). The `torch>=2.0` line in `requirements.txt` may need to be satisfied via [pytorch.org](https://pytorch.org) instructions for GPU wheels.

- Put the package on the path:

  ```bash
  export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
  ```

  Or install the package in editable mode if you add a `pyproject.toml` / `setup.cfg`; until then, `PYTHONPATH=src` is the usual approach.

- **Checkpoint:** use **`models/CNN_best_model.pt`** from the clone (~2 MB) or your own `final_models/CNN_best_model.pt` from a training run.

- **Jupyter:** open [`notebooks/evoCAST_spacer_Colab_v2.ipynb`](notebooks/evoCAST_spacer_Colab_v2.ipynb) from the repo root; set `REPO_DIR` / paths to your clone instead of `/content/...` if you adapt the setup cell.

## Making predictions (CNN, genomic FASTA)

This flow scores **candidate protospacers** in a genomic sequence using a saved **CNN** checkpoint (`save_torch_best_checkpoint`). The repository includes a released weights file at **`models/CNN_best_model.pt`** (~2.2 MB; optimized retrain from the training pipeline) so you can run inference without rebuilding `output/modeling/.../final_models/`.

### End-to-end workflow (inference → ranked spacers → picking → primers → donor plasmids)

Run steps **in order** from the repo root with `PYTHONPATH=src` (or your usual `export`). The Colab notebook runs the same logic interactively (see **Quick start** above).

1. **Inference (scan + extract + score)** — `spacerModel.inference.genomic_spacer_predict_cnn` walks each FASTA record for **YN-PAM** sites, **extracts** 32 bp spacers on both strands, and scores them with the CNN. It writes **`ranked_sequences.xlsx`** (all genes) and **`ranked_sequences_<gene>.xlsx`** per gene in the **directory containing** your `--output` argument (see CLI below). Those tables are the **ranked spacer** inputs for the next steps.

2. **Picking (curated sets + Primer-BLAST)** — `spacerModel.inference.picking` reads a ranked table (typically `ranked_sequences.xlsx` or a per-gene workbook) plus the **same reference FASTA** used in step 1. For each gene it can take the **best**-ranked spacers, optionally **worst**-ranked, and **random** picks from the rest, and runs **NCBI Primer-BLAST** (per spacer) to attach primer pairs. After picking, each selected row also gets:
   - **`unique_in_locus`**: `True` if a 1-nt sliding scan of the gene FASTA finds exactly one perfect match to that spacer (or its reverse complement).
   - **`no_similar_spacers`**: `True` if no *other* ranked candidate for the same gene shares **more than 24** identical nucleotides (fixed SpacerSeq alignment; self excluded).

   Example:

   ```bash
   PYTHONPATH=src python -m spacerModel.inference.picking \
     --input output/inference/ranked_sequences.xlsx \
     --fasta path/to/same_genome.fa \
     --output output/inference/picked_spacers.xlsx
   ```

   Use `--number-top-picked`, `--number-bottom-picked`, `--number-random-picked`, and `--gene` (single gene or `all`) as needed; see `--help`.

3. **Primer search only (optional)** — If you want Primer-BLAST on **every row** of a ranked table **without** the picking logic, use `spacerModel.inference.primer` (see below). Picking already performs primer search for the rows it selects; use `primer.py` when you need a fully annotated ranked sheet first.

4. **Donor plasmid constructs (optional)** — `src/cloning/func_automated_cloning.py` inserts a spacer sequence into the PseCAST donor template between fixed flanking motifs and writes GenBank with a **`spacer`** feature. Template: **`data/PseCAST-exampleDonor_template-pSE1405.gbk`**. In the Colab notebook this runs automatically for the **top pick per gene** (first row with a passing primer pair, or top model rank if primers were skipped). Example (Python):

   ```python
   from pathlib import Path
   from cloning.func_automated_cloning import insert_sequence_with_feature

   insert_sequence_with_feature(
       "data/PseCAST-exampleDonor_template-pSE1405.gbk",
       "output/donor_plasmids/PseCAST_MYGENE_rank1_donor.gbk",
       flank_5_prime="GTGACCTGCCGTATAGGCAGCTGAAAAT",
       flank_3_prime="GTGACCTGCCGTATAGGCAGC",
       insert="ACGT...",  # 32 bp spacer
       feature_label="spacer",
   )
   ```

   See [`src/cloning/cloning_example.py`](src/cloning/cloning_example.py) for batch usage patterns.

**What you need**

- A checkpoint file (`.pt`) with `model_kind == "cnn"`, trained on **flat one-hot** DNA (`--embedding onehot`), consistent with how the model was saved. Use **`models/CNN_best_model.pt`** from the clone or your own `final_models/CNN_best_model.pt` from a training run.
- A **FASTA** file (e.g. `.fa`). Each ``>`` header line is stored **verbatim** (text after ``>``, stripped) in the output **Gene** column. For **picking** (`--gene`) and **per-gene density plots**, the matchable “symbol” is the substring **before the first underscore** when the header contains `_` (e.g. ``>MYO7A_chr7_...`` → ``MYO7A``). If there is **no** underscore, the whole header is used. Underscores are **not** required; they only help separate a short gene tag from extra text in the header. Plain text **without** any ``>`` line is treated as a single sequence with ``Gene`` set to ``(no_header)``.

**Convention**

- Spacer length is **32 bp** by default; it must match the checkpoint (`n_features = 4 × length`).
- Sites are scanned on **both strands** for a **YN** PAM (first base Y ∈ {C, T}) immediately 5′ of the spacer: `[PAM(2)][SPACER(32)]` on the targeting strand. Output coordinates are **1-based** on the forward reference.
- **Flanking sequence (default 400 bp each end):** Each FASTA record should extend **at least ~400 nt upstream and ~400 nt downstream** beyond the genomic interval where you want protospacers scored. By default the scanner **skips** the outer **400 bp** on each end of the record (edge reserve) so hits stay away from margins while **coordinates stay 1-based on the full record** (for primer tools). If your sequence is shorter than ``2 × edge_reserve``, that record is skipped. For a **full-length** scan of the record, use ``--no-omit-edge-bases`` (or ``--spacer-search-edge-reserve 0``). See ``--help``.

**CLI**

From the repository root (with `PYTHONPATH=src`):

```bash
python -m spacerModel.inference.genomic_spacer_predict_cnn \
  --checkpoint models/CNN_best_model.pt \
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

- **In-silico cloning utilities**: `src/cloning/func_automated_cloning.py` (GenBank / SnapGene `.dna` → insert between flanks + feature annotation)
- **Dataset analysis and logos**: `src/spacerModel/data/analyze_dataset.py`
- **Simple hypothesis tests on data**: `src/spacerModel/simple_modeling/simple_hypothesis_test.py`
- **Training / hyperparameter search**: see `src/spacerModel/training/` (`train_torch.py`, `torch_hyperopt.py`, `retrain_optimized_models.py`, etc.).
- **CNN explainability** (saliency, layer activations): `src/spacerModel/interpretation/explain_cnn_layers.py`.
