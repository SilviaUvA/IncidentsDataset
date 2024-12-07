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
# python3 run_model.py --config configs/multi_label_final_model --mode test --activation softmax --images_path data/images --dataset_train data/multi_label_train_clean.json --dataset_val data/multi_label_val_clean.json --dataset_test data/multi_label_val_clean.json
# python3 run_model.py --config configs/multi_label_final_model --mode train --images_path data/images


# python3 run_model.py --num_gpus 1 --dataset pos_and_neg --config configs/multi_label_final_model --mode test --activation sigmoid --images_path data/images --dataset_train data/test.json --dataset_val data/test.json --dataset_test data/test.json



#TODO check whether it should be pos_and_neg or pos_only... -> pos_and_neg bcs we want to include images with no disaster
#TODO can make num_incidents == 1 for binary + --activation softmax
#TODO --pretrained does not use a pretrained model
#TODO check how to do binary classification and adjust the labels in the code for it??
python3 run_model.py --binary --ignore_places --pretrained --pretrained_with_places False --arch resnet18 --dataset pos_and_neg --config configs/baseline_model_binary --mode train --activation softmax --images_path data/images_eccv #--dataset_train data/multi_label_val_clean.json --dataset_val data/test.json --dataset_test data/test.json


# python3 run_model.py --num_gpus 1 --arch resnet18 --dataset pos_and_neg --config configs/multi_label_final_model --mode test --activation sigmoid --images_path data/images --dataset_train data/test.json --dataset_val data/test.json --dataset_test data/test.json
# python3 run_model.py --num_gpus 1 --arch resnet18 --dataset pos_only --config configs/multi_label_final_model_class_positive_only --mode test --activation sigmoid --images_path data/images --dataset_train data/test.json --dataset_val data/test.json --dataset_test data/test.json
