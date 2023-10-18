#!/bin/bash
# ====================================
#SBATCH --job-name=CNNProt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=32G               # Request the full memory of the node
#SBATCH --time=12:00:00
#SBATCH --account=def-xinwang1
# ====================================

python cnn2.py --embedding_size 21 --w2v_epochs 20 --num_gpus 1
