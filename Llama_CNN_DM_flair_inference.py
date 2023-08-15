import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from flair.data import Sentence
from flair.models import SequenceTagger
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

    print('set tagger..')
    tagger = SequenceTagger.load("flair/ner-english-ontonotes")
    print('done..')

    train_dataset = dataset["train"]
    tokenizer.add_special_tokens(
        {
            
            "pad_token": "<PAD>",
        }
    )

    test_sampled = dataset["test"].shuffle(seed=42).select(range(1000))
    # test_sampled = dataset["test"]

    # tokenizer.pad_token = tokenizer.eos_token
    prompt = "\nlet's integrate the above information and summarize the above article:\n"
    full_text_sampled = []
    for i in test_sampled['article']:
        sentence = Sentence(i)
        tagger.predict(sentence)
        person = []
        date = []
        event = []
        org = []
        for label in sentence.get_labels('ner'):
            if label.value == 'PERSON':
                person.append(label.data_point.text)
            elif label.value == 'DATE':
                date.append(label.data_point.text)
            elif label.value == 'EVENT':
                event.append(label.data_point.text)
            elif label.value == 'ORG':
                org.append(label.data_point.text)
        if person:
            person = 'The people in this document are ' + ', '.join(s for s in person) + '\n'
        else:
            person = ""
        if date:
            date = 'The dates in this document are ' + ', '.join(s for s in date) + '\n'
        else:
            date = ""
        if event:
            event = 'The events in this document are' + ', '.join(s for s in event) + '\n'
        else:
            event = ""
        if org:
            org = 'The organizations in this document are' + ', '.join(s for s in org) + '\n'
        else:
            org = ""
        ner_prompt = person + date + event + org
        article_prompt = i + '\n' + ner_prompt + prompt
        full_text_sampled.append(article_prompt)

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

    with open('results_llama_7n_cnndm_test', 'w') as f:
        f.writelines(summaries)
    print('write data')
    references = test_sampled['highlights']
    print('compute results..')
    print(metric.compute(predictions=summaries, references=references))

    print('끝')
if __name__=='__main__':
    main()
