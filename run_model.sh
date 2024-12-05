#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=HCML
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=runModel_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source ~/.bashrc
cd $HOME/IncidentsDataset

conda activate hcml

# download weights before running model
# python3 run_download_weights.py #uncomment if need to download weights still

# run the model
# python3 run_model.py --config configs/multi_label_final_model --mode test --activation softmax --images_path /data/images --dataset_train data/multi_label_train_clean.json --dataset_val data/multi_label_val_clean.json --dataset_test data/multi_label_val_clean.json
# python3 run_model.py --config configs/multi_label_final_model --mode train --images_path /data/images
python3 run_model.py --config configs/multi_label_final_model --mode test --activation softmax --images_path /data/images --dataset_train data/multi_label_val.json --dataset_val data/multi_label_val.json --dataset_test data/multi_label_val.json
