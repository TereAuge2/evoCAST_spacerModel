# --- Git / GitHub (run from repo root: 2603_evoCAST_spacer_model) ---
# Remote (SSH): git@github.com:TereAuge2/evoCAST_spacerModel.git
#   HTTPS URL (same repo): https://github.com/TereAuge2/evoCAST_spacerModel
# First-time clone:  git clone git@github.com:TereAuge2/evoCAST_spacerModel.git
# Day to day:
#   git status
#   git add -A
#   git commit -m "Short message describing the change"
#   git push origin main
# If push asks for a password over HTTPS, use SSH remote instead (see above) or a GitHub PAT.
# Do not commit secrets: .wandb_env is gitignored; copy from .wandb_env.example if needed.
# Large/generated dirs are ignored (see .gitignore): output/, logs/, wandb/

# Quick import / CLI checks (run in an interactive session with evoCAST; PYTHONPATH=src):
#   python -m spacerModel.training.train_torch --help
#   python -m spacerModel.training.torch_hyperopt --help

# Requesting a single core for an interactive job for 1 hour
salloc -t 00:30:00 -p mit_normal

module load miniforge
mamba activate evoCAST


# Data analysis:

python src/spacerModel/data/analyze_dataset.py 

python src/spacerModel/data/analyze_dataset.py --splits-excel output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx


# logos
export PYTHONPATH=src
python -m spacerModel.interpretation.make_cluster_logo_movie \
  --logo-dir output/data_processing/logos \
  --which by_locus \
  --fps 2

#"all", "clusters", "splits", "by_locus"
export PYTHONPATH=src
python -m spacerModel.interpretation.make_cluster_logo_movie \
  --logo-dir output/data_processing/logos \
  --which clusters 

export PYTHONPATH=src
python -m spacerModel.interpretation.make_cluster_logo_movie \
  --logo-dir output/data_processing/logos \
  --which splits \
  --fps 2


## Linear position weights
python src/spacerModel/interpretation/linear_position_weights.py \
  --excel output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx \
  --split-col Split \
  --embeddings onehot,kmer_gc_onehot \
  --kmer-k 2 \
  --output-dir output/interpretation/linear_position_weights


#### Train models
sbatch src/spacerModel/scripts/orcd_train_models_array.sbatch



# Default embedding is kmer_gc_onehot + dense (--raw-onehot default); override RUN_NAME as needed.
sbatch --export=ALL,RUN_NAME=Split__kmer_gc_onehot_k2_default src/spacerModel/scripts/orcd_train_simple_models.sbatch

sbatch --export=ALL,EMBEDDING=onehot,RUN_NAME=Split__onehot_raw_CNN_only,EXTRA_ARGS='--models cnn1d' src/spacerModel/scripts/orcd_train_simple_models.sbatch

python src/spacerModel/simple_modeling/train_models.py --split-col Split

python src/spacerModel/simple_modeling/train_models.py --split-col SplitRandom

PYTHONPATH=src python src/spacerModel/simple_modeling/combine_cv_summaries.py


#replotting with median baseline

export PYTHONPATH=src
python -m spacerModel.simple_modeling.replot_cv_summaries \
  --input-base output/modeling \
  --run-dirs \
    "0_Split__kmer_gc_onehot_raw__k2__A11038783_T0__20260326_204925" \
    "0_SplitRandom__kmer_gc_onehot_raw__k2__A11039227_T0__20260326_210821" \
    "1_Best_initial_CV_pre_hyperopt_Split__kmer_gc_onehot_raw__k2_manual" \
  --median-baseline
  
#Plotting (old scripts)
python src/spacerModel/simple_modeling/replot_cv_summaries.py \
  --input-base output/modeling \
  --run-dirs "90%_variance_SVD" "100%_variance_SVD" raw_onehot



# Hyperparameter tuning — CNN smoke test (N_ITER=5) + wandb
#   export WANDB_API_KEY=...   # once

# Full torch hyperopt on CPU (W&B on by default; USE_WANDB=0 to skip):
sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_cnn_cpu.sbatch
sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_mlp_cpu.sbatch

#SVR
sbatch src/spacerModel/scripts/orcd_tune_svr_kernel.sbatch

#XGboost
sbatch src/spacerModel/scripts/orcd_tune_xgboost.sbatch

#old - plot hyperparameter
python src/spacerModel/training/plot_hyperopt_predictions.py \
  --run-dir output/modeling/CNN_hyperparameter_tune/11074812_20260327_162051


#explainaiblity on CNN best model
N_PER_GROUP=300 sbatch src/spacerModel/scripts/orcd_explain_cnn_layers.sbatch \
  --checkpoint "models/CNN_best_model.pt"


#inference on new data
PYTHONPATH=src python -m spacerModel.inference.genomic_spacer_predict_cnn \
  --checkpoint "models/CNN_best_model.pt" \
  --fasta data/input_fasta_edge.fa \
  --output output/inference/ranked_spacers.xlsx 


## getting primers for all genes


## searching per spacer per gene --> only top 3 spacers per gene 
PYTHONPATH=src python -m spacerModel.inference.primer \
  --input output/inference/ranked_sequences_MYO7A_round2__hg19_knownGene_ENST00000962572.1_1_1stIntron.xlsx \
  --ncbi-num-primer-pairs 50 \
  --limit 2 \
  --ncbi-html-dir output/inference/ncbi_html \
  --fasta data/input_fasta_edge.fa \
  --output output/inference/primers_top_5_MYO7A_redo3.xlsx \
  --per-spacer-template \
  --verbose

## randomly pick and pick top 3 from model and combine with Simon (only takes the top 3 of model with primers)
# this times out on interactive session
PYTHONPATH=src python -m spacerModel.inference.picking \
  --input output/inference/ranked_sequences.xlsx \
  --fasta data/input_fasta_edge.fa \
  --output output/inference/Model_ranked_spacers_all_genes_picked_for_testing_with_primers.xlsx \
  --gene all \
  --number-top-picked 3 \
  --number-bottom-picked 3 \
  --number-random-picked 3 \
  --ncbi-num-primer-pairs 50 \
  --output-ranked output/inference/Model_ranked_spacers_all_genes.csv

# Long runs: submit from repo root, e.g.
  sbatch src/spacerModel/scripts/orcd_picking_all_genes.sbatch
# Override paths: sbatch --export=ALL,PICK_INPUT=...,PICK_FASTA=...,PICK_BOTTOM=0,...


#CNN surrogate XGBoost
sbatch src/spacerModel/scripts/orcd_cnn_surrogate_xgboost.sbatch \
  --shap-contrib-max-rounds 400 \
  --shap-synthetic-max-samples 2048

PYTHONPATH=src python -m spacerModel.interpretation.replot_surrogate_shap \
  --output-dir output/interpretation/cnn_surrogate_xgboost \
  --stems synthetic,real_cv