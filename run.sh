#!/bin/bash

#SBATCH -J test
#SBATCH -o output/test.%j.out
#SBATCH -p A6000
#SBATCH --gres=gpu:3
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

python llm_sum.py --model 'meta-llama/Llama-2-7b-hf' \
    --dataset 'cnn_dailymail' \
    --do_test \
    --epoch 1 \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --saving_steps 5000 \
    --test_length 10 \
    --output_model_dir './output/test/' \
    --save_summary \
    --output_summary_dir './result/test/' \
    --openai_API_key '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' \