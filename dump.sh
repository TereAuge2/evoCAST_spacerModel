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
salloc -t 00:10:00 -p mit_normal

srun --pty -t 01:00:00 --mem=4G -c 2 bash


module load miniforge
mamba activate evoCAST


#### Train models
sbatch src/spacerModel/scripts/orcd_train_models_array.sbatch



# Default embedding is kmer_gc_onehot + dense (--raw-onehot default); override RUN_NAME as needed.
sbatch --export=ALL,RUN_NAME=Split__kmer_gc_onehot_k2_default src/spacerModel/scripts/orcd_train_simple_models.sbatch

sbatch --export=ALL,EMBEDDING=onehot,RUN_NAME=Split__onehot_raw_CNN_only,EXTRA_ARGS='--models cnn1d' src/spacerModel/scripts/orcd_train_simple_models.sbatch

python src/spacerModel/simple_modeling/train_models.py --split-col Split

python src/spacerModel/simple_modeling/train_models.py --split-col SplitRandom


#Plotting
python src/spacerModel/simple_modeling/replot_cv_summaries.py \
  --input-base output/modeling \
  --run-dirs "90%_variance_SVD" "100%_variance_SVD" raw_onehot

# Data analysis:
python src/spacerModel/interpretation/linear_position_weights.py \
  --excel output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx \
  --split-col Split \
  --embeddings onehot,kmer_gc_onehot \
  --kmer-k 2 \
  --output-dir output/interpretation/linear_position_weights


python src/spacerModel/data/analyze_dataset.py 

python src/spacerModel/data/analyze_dataset.py --splits-excel output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx



# Hyperparameter tuning — CNN smoke test (N_ITER=5) + wandb
#   export WANDB_API_KEY=...   # once
# CPU (mit_normal): use if GPU queue is stuck
sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_cpu_test.sbatch
# GPU (mit_normal_gpu): sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_gpu_test.sbatch

# Full torch hyperopt on CPU (W&B on by default; USE_WANDB=0 to skip):
sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_cnn_cpu.sbatch
sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_mlp_cpu.sbatch

#SVR
sbatch src/spacerModel/scripts/orcd_tune_svr_kernel.sbatch

#XGboost
sbatch src/spacerModel/scripts/orcd_tune_xgboost.sbatch

#plot hyperparameter
python src/spacerModel/training/plot_hyperopt_predictions.py \
  --run-dir output/modeling/CNN_hyperparameter_tune/11074812_20260327_162051