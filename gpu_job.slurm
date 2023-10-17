#!/bin/bash
#! /bin/bash
# ====================================
#SBATCH --job-name=CNNProt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-v100
# ====================================

python cnn2.py
