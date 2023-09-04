import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import default_data_collator, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import evaluate
from datasets import load_dataset

from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import cnn_dm_dataset
from unieval_utils import convert_to_json
from metric.evaluator import get_evaluator

import openai
import argparse
import nltk
import time
import json
import math

from datasets import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate summarization models")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Summarization Model name'
    )
    parser.add_argument(
        '--flair',
        action='store_true',
        help='Add Flair model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        default='cnn_dailymail',
        help='Summarization dataset'
    )
    parser.add_argument(
        '--do_train',
        action='store_true',
        help='Train model'
    )
    parser.add_argument(
        '--do_test',
        action='store_true',
        help='Test model'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        required=False,
        default=10,
        help='Epoch'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        required=False,
        default=4,
        help='Train batch size'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        required=False,
        default=4,
        help='Test batch size'
    )
    parser.add_argument(
        '--saving_steps',
        type=int,
        required=False,
        default=5000,
        help='Saving steps'
    )
    parser.add_argument(
        '--test_length',
        type=int,
        required=True,
        help='How much data to test'
    )
    parser.add_argument(
        '--output_model_dir',
        type=str,
        required=True,
        help='The name of trained model'
    )
    parser.add_argument(
        '--save_summary',
        action='store_true',
        help='Whether save summary or not'
    )
    parser.add_argument(
        '--output_summary_dir',
        type=str,
        required=True,
        help='Directory of generated summaries'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Push the trained model on huggingface or not'
    )
    parser.add_argument(
        '--openai_API_key',
        type=str,
        required=False,
        help='openAI API key'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        required=False,
        help='openAI API key'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f'{args.model} is used')

    start = time.time()

    model_ckpt = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{model_ckpt} is running on {device}..\n")
    print('load dataset..')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, version='3.0.0')
        if args.model == 'meta-llama/Llama-2-7b-hf':
            tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)
        train_dataset = get_preprocessed_dataset(tokenizer, cnn_dm_dataset, 'train')
        test_sampled = dataset["test"].shuffle(seed=42).select(range(args.test_length))
    if args.dataset == 'element_aware':
        with open('./ft_datasets/cnndm_element_aware.json') as f:
            dataset = json.load(f)
        articles = []
        references = []
        for i in dataset['cnn_dm']:
            articles.append(i['src'])
            references.append(i['element-aware_summary'])
    print('done..')
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
    
    print(args)
    if args.do_train :
        print('train model.2.')
    if args.model == 'meta-llama/Llama-2-7b-hf':
        model =LlamaForCausalLM.from_pretrained(model_ckpt, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        if args.do_train:
            model.train()
            model, lora_config = create_peft_config(model)

            output_dir = f'{args.output_model_dir}/epoch{args.epoch}_batch{args.train_batch_size}'
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=args.epoch,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.test_batch_size,
                bf16=True,
                logging_dir=f"{output_dir}/logs",
                logging_strategy="steps",
                logging_steps=10,
                save_strategy="steps",
                save_steps=args.saving_steps,
                optim="adamw_torch_fused",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=default_data_collator,
            )
            trainer.train()
            print('done')
            if args.push_to_hub == True:
                trainer.push_to_hub()
                print('push model to hub')

        if args.do_test:
            prompt = '\n\nSummarize the above article:\n'
            full_text_sampled = []
            for i in test_sampled['article']:
                full_text_sampled.append('Article:\n' + i + prompt)
            print('make full data')
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
                    output = model.generate(input_ids=input_ids_sampled[i].to(device), attention_mask=attention_mask_sampled[i].to(device), max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, length_penalty=2)
                    predictions_all = tokenizer.decode(output[0], skip_special_tokens=True)
                    predictions = predictions_all.split(prompt)[1]
                    print(f'{i}th summary:')
                    print(predictions)
                    predictions = [p.replace('\n', '/') for p in predictions]
                    print('\n')
                    summaries.append(predictions)
            references = test_sampled['highlights']
            metric = evaluate.load('rouge', use_aggregator=False)
            print(metric.compute(predictions=summaries, references=references))

    if args.model == 'facebook/bart-base':
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

        def chunks(list_of_elements, batch_size):
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i: i+batch_size]
        
        def evaluate_summary(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"):
            article_batches = list(chunks(dataset[column_text], batch_size))
            target_batches = list(chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
                inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                                padding="max_length", return_tensors="pt")
                summaries = model.generate(input_ids=inputs["input_ids"].to(device), # module을 추가하면 dataparallel문제 해결
                                        attention_mask=inputs["attention_mask"].to(device),
                                        length_penalty=0.8, num_beams=8, max_length=128)
                decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True) for s in summaries]
                decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
                print(decoded_summaries)
                metric.add_batch(predictions=decoded_summaries, references=target_batch)

            score = metric.compute()
            return score

        def convert_examples_to_features(example_batch):
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            input_encodings = tokenizer(example_batch["article"], max_length=1024, truncation=True)
            with tokenizer.as_target_tokenizer():
                target_encodings = tokenizer(example_batch["highlights"], max_length=128, truncation=True)
            return {"input_ids": input_encodings["input_ids"],
                    "attention_mask": input_encodings["attention_mask"],
                    "labels": target_encodings["input_ids"]}
        print('load dataset..')
        dataset_pt = dataset.map(convert_examples_to_features, batched=True)
        columns = ["input_ids", "labels", "attention_mask"]
        dataset_pt.set_format(type="torch", columns=columns)
        print('done..')
        if args.do_train:
            seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            training_args = TrainingArguments(
                output_dir=args.output_model_dir,
                num_train_epochs=args.epoch,
                warmup_steps=500,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.test_batch_size,
                weight_decay=0.01,
                logging_steps=10,
                push_to_hub=False,
                evaluation_strategy='steps',
                eval_steps=500,
                save_steps=1e6,
                remove_unused_columns=False,
                gradient_accumulation_steps=16
            )

            trainer = Trainer(model=model, args=training_args,
                        tokenizer=tokenizer,
                        data_collator=seq2seq_data_collator,
                        train_dataset=dataset_pt['train'],
                        eval_dataset=dataset_pt['validation'])
            print('train model..1')
            trainer.train()
            print('done')
            if args.push_to_hub == True:
                trainer.push_to_hub()
                print('push model to hub')
        if args.do_test:
            score = evaluate_summary(test_sampled, metric, trainer.model, tokenizer, batch_size=2, column_text="article", column_summary="highlights")
            results = trainer.evaluate()
            print(f'result: {results}')
            print(f'score: {score}')

    if args.model == 'gpt-4':
        if args.do_test:
            openai.api_key = args.openai_API_key
            summarization_prompt = "\nSummarize the above article:\n"
            system_prompt = "You are expert at summarizing, and you are going to summarize some articles"

            chat_response = []

            for idx, text in enumerate(test_sampled['article']):
                try:
                    summary = openai.ChatCompletion.create(model='gpt-4', messages=[
                        # {'role': 'system', 'content': system_prompt}, 
                        {'role': 'user', 'content': summarization_prompt+text}
                    ])
                    chat_response.append(summary)
                    print(f'idx: {idx}\n')
                    print(summary)
                    sleep(15)
                except:
                    print('error')
                    break
            
        chat_predictions = []
        for data in chat_response:
            summary = data['choices'][0]['message']['content']
            chat_predictions.append(summary)
            print(summary)
        print(metric.compute(predictions=chat_predictions, references=test_sampled[:len(chat_predictions)]['highlights']))

    
    if args.do_test:
        nltk.download('punkt')
        task = 'summarization'

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=summaries, 
                        src_list=articles, ref_list=references)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, print_result=True)
        print('eval score:', eval_scores)


    if args.save_summary == True:
        with open(f'{args.output_summary_dir}/{args.model}_epoch{args.epoch}_len{args.test_length}' 'w') as f:
            f.writelines(summaries)

    end = time.time()  
    print(f"time used: {end-start:.2f}sec")

    
if __name__ == "__main__":
    main()
