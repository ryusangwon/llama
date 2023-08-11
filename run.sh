#!/bin/bash

#SBATCH -J llama_lora_CNN_DM
#SBATCH -o output/llama_lora_CNN_DM.%j.out
#SBATCH -p A5000
#SBATCH --gres=gpu:8
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

python Llama_Lora_CNN_DM.py