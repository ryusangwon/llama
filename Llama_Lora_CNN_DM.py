import torch
from datasets import load_dataset, load_metric, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import default_data_collator, Trainer, TrainingArguments

from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import cnn_dm_dataset

import evaluate

import nltk
import numpy as np
import pandas as pd



def main():
    print('start')
    print('set dataset..')
    dataset = load_dataset('cnn_dailymail', version='3.0.0')
    metric = evaluate.load('rouge', use_aggregator=False)
    print('done..')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('set model..')
    model_ckpt="meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)
    model =LlamaForCausalLM.from_pretrained(model_ckpt, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    print('done..')

    train_dataset = dataset["train"]
    tokenizer.add_special_tokens(
        {
            
            "pad_token": "<PAD>",
        }
    )

    print('prepare data..')
    train_dataset = get_preprocessed_dataset(tokenizer, cnn_dm_dataset, 'train')
    print('prepare data done..')

    
    model.train()

    def create_peft_config(model):
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            prepare_model_for_int8_training,
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )

        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    # create peft config
    model, lora_config = create_peft_config(model)

    print('train model..')
    output_dir = './output/'
    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()
    print('train model done..')
        
    print('start evaluation..')
    test_sampled = dataset["test"].shuffle(seed=42).select(range(1000))

    full_text_sampled = []
    for i in test_sampled['article']:
        full_text_sampled.append(i + '\nSummarize this article:\n')
    print('make full data..')
    article_sampled = Dataset.from_pandas(pd.DataFrame(data=full_text_sampled, columns=['article']))
        
    def tokenize(examples):
        output = tokenizer(examples['article'], return_tensors='pt').to(device)
        return output
    
    tokenized_article_sampled = article_sampled.map(tokenize)
    print('tokenize dataset')
    
    input_ids_sampled = tokenized_article_sampled.with_format('torch')['input_ids']
    attention_mask_sampled = tokenized_article_sampled.with_format('torch')['attention_mask']


    print('start make summaries..')
    summaries = []
    for i in range(len(tokenized_article_sampled)):
        with torch.no_grad():
            output = model.generate(input_ids=input_ids_sampled[i].to(device), attention_mask=attention_mask_sampled[i].to(device), max_length=2048, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, length_penalty=2)
            predictions_all = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions = predictions_all.split('Summarize this article:\n')[1]
            print(f'{i}th summary:')
            print(predictions_all)
            print('\n')
            summaries.append(predictions)
    print(f"길이 {len(summaries)}")
    print('done..')
    print('write summaries..')
    with open('results_llama_7b_lora_cnndm', 'w') as f:
        f.writelines(summaries)
    print('done')
    references = test_sampled['highlights']
    print('compute results..')
    print(metric.compute(predictions=summaries, references=references))

    print('evalute done..')
if __name__=='__main__':
    main()
