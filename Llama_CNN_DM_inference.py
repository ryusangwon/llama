import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import evaluate
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset

def main():
    print('start')
    dataset = load_dataset('cnn_dailymail', version='3.0.0')
    metric = evaluate.load('rouge', use_aggregator=False)

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

    test_sampled = dataset["test"].shuffle(seed=42).select(range(1000))
    test_sampled = dataset["test"]

    # tokenizer.pad_token = tokenizer.eos_token
    prompt = '\nSummarize the above article:\n'
    full_text_sampled = []
    for i in test_sampled['article']:
        full_text_sampled.append(i + prompt)
    print('make full data')
    article_sampled = Dataset.from_pandas(pd.DataFrame(data=full_text_sampled, columns=['article']))
        
    def tokenize(examples):
        output = tokenizer(examples['article'], return_tensors='pt')
        return output
    
    tokenized_article_sampled = article_sampled.map(tokenize)
    print('tokenize dataset')
    
    input_ids_sampled = tokenized_article_sampled.with_format('torch')['input_ids']
    attention_mask_sampled = tokenized_article_sampled.with_format('torch')['attention_mask']

    print('start make summaries..')
    summaries = []
    for i in range(len(tokenized_article_sampled)):
        with torch.no_grad():
            output = model.generate(input_ids=input_ids_sampled[i].to(device), attention_mask=attention_mask_sampled[i].to(device), max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, length_penalty=2)
            predictions_all = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions = predictions_all.split(prompt)[1]
            print(f'{i}th summary:')
            print(predictions_all)
            print('\n')
            summaries.append(predictions)
    print(f"길이 {len(summaries)}")

    with open('results_llama_7n_cnndm', 'w') as f:
        f.writelines(summaries)
    print('write data')
    references = test_sampled['highlights']
    print('compute results..')
    metric.compute(predictions=summaries, references=references)

    print('끝')
if __name__=='__main__':
    main()
