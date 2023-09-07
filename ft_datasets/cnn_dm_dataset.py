# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/cnn_dailymail

import datasets
from .utils import Concatenator

def get_preprocessed_cnn_dm(dataset_config, tokenizer, split):
    # dataset = datasets.load_dataset("cnn_dailymail", version='3.0.0', split=split)
    dataset = datasets.load_dataset("cnn_dailymail", version='3.0.0', split=split).shuffle(seed=42).select(range(10000))

    prompt = (
        f"{{article}}\nSummarize this article:\n{{highlights}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                article=sample["article"],
                highlights=sample["highlights"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
