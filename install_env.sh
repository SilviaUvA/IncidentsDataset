#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=install_env_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source ~/.bashrc
cd $HOME/IncidentsDataset

conda remove -n hcml --all #TODO remove later when no more packages are necessary to be installed
conda create -n hcml python=3.8.2
conda activate hcml
pip install -r requirements.txt