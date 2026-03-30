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

#other models:
sbatch --export=ALL,MODEL_KIND=cnn,RUN_NAME=CNN_hyper_test src/spacerModel/scripts/orcd_train_torch_hyperopt.sbatch
sbatch --export=ALL,MODEL_KIND=mlp,RUN_NAME=MLP_hyper_test src/spacerModel/scripts/orcd_train_torch_hyperopt.sbatch

#SVR
sbatch --export=ALL,MODEL_KIND=svr,RUN_NAME=SVR_hyper_test src/spacerModel/scripts/orcd_train_torch_hyperopt.sbatch


#plot hyperparameter

python -m src.spacerModel.models.plot_hyperopt_predictions \
  --run-dir output/modeling/CNN_hyperparameter_tune/11074812_20260327_162051