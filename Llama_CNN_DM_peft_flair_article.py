import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
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


    peft_model_id = "./output/10000dataset3epoch"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    print('set tagger..')
    tagger = SequenceTagger.load("flair/ner-english-ontonotes")
    print('done..')

    train_dataset = dataset["train"]
    tokenizer.add_special_tokens(
        {
            
            "pad_token": "<PAD>",
        }
    )

    test_sampled = dataset["test"].shuffle(seed=42).select(range(10))
    # test_sampled = dataset["test"]

    # tokenizer.pad_token = tokenizer.eos_token
#     entity_prompt = """Entity Instruction:
# The entities in the article are mentioned between '[' and ']' for what entity the words are.
# example)
# [Person] is person
# [Organization] is organization
# ['Date'] is date
# ['Event'] is event\n\n"""
    entity_prompt = """Entity Instruction:
The entities in the article are mentioned between '[' and ']' for what entity the words are. Entity categories are person, date, event, organization.\n\n"""

    prompt = "\nSummarize the above article:\n"
    full_text_sampled = []
    for article in test_sampled['article']:
        sentence = Sentence(article)
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

        person = {'person': list(set(person))}
        date = {'date': list(set(date))}
        event = {'event': list(set(event))}
        org = {'organization': list(set(org))}

        for per in list(person.values())[0]:
            article = article.replace(per, per + '[' + list(person.keys())[0] + ']')
        for da in list(date.values())[0]:
            article = article.replace(da, da + '[' + list(date.keys())[0] + ']')
        for eve in list(event.values())[0]:
            article = article.replace(eve, eve + '[' + list(event.keys())[0] + ']')
        for o in list(org.values())[0]:
            article = article.replace(o, o + '[' + list(org.keys())[0] + ']')
        
        article_prompt = entity_prompt + 'article:\n' + article + '\n\n' + prompt
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
            output = model.generate(input_ids=input_ids_sampled[i].to(device), attention_mask=attention_mask_sampled[i].to(device), max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, length_penalty=2.0)
            predictions_all = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions = predictions_all.split(prompt)[1]
            print(f'{i}th:')
            print(predictions_all)
            print('\n')
            summaries.append(predictions)
    print(f"길이 {len(summaries)}")

    with open('results_llama_7n_cnndm_peft_flair', 'w') as f:
        f.writelines(summaries)
    print('write data')
    references = test_sampled['highlights']
    print('compute results..')
    print(metric.compute(predictions=summaries, references=references))

    print('끝')
if __name__=='__main__':
    main()
