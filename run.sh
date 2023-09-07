#!/bin/bash

#SBATCH -J llama
#SBATCH -o output/llama_rep12_len20_prompt별_1000개%j.out
#SBATCH -p A5000
#SBATCH --gres=gpu:1
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
    --test_size 1000 \
    --num 12 \
    --output_model_dir './output' \
    --save_summary \
    --output_summary_dir './result' \
    --openai_API_key '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' \

# python llm_sum.py --model "ryusangwon/bart-cnndm" \
#     --dataset 'cnn_dailymail' \
#     --do_test \
#     --epoch 1 \
#     --train_batch_size 4 \
#     --test_batch_size 4 \
#     --saving_steps 5000 \
#     --test_size 10 \
#     --num 12 \
#     --output_model_dir './output' \
#     --output_summary_dir './result' \
#     --openai_API_key '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' \


# python llm_sum.py --model "ryusangwon/bart-cnndm" \
#     --dataset 'cnn_dailymail' \
#     --do_test \
#     --epoch 1 \
#     --train_batch_size 4 \
#     --test_batch_size 4 \
#     --saving_steps 5000 \
#     --test_size 10 \
#     --num 12 \
#     --output_model_dir './output' \
#     --save_summary \
#     --output_summary_dir './result' \
#     --openai_API_key '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' \