#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=HCML
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --output=runDownloadData_%A.out
#SBATCH --error=errorDownloadData_%A.out


module purge
module load 2024
module load Anaconda3/2024.06-1

source ~/.bashrc
cd $HOME/IncidentsDataset

conda activate hcml

python3 download_data.py
# download weights before running model
# python3 run_download_weights.py #uncomment if need to download weights still

# run the model
# python3 run_model.py --config configs/multi_label_final_model --mode val --activation softmax --images_path /data/images
# python3 run_model.py --config configs/multi_label_final_model --mode train --images_path /data/images
