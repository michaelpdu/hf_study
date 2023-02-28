#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : mlm.py
# @Time    : 2022/12/17 19:20
# @Author  : dupei.gs.njust@gmail.com
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


if __name__ == "__main__":
    model_checkpoint = "model/distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    distilbert_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
    print(f"'>>> BERT number of parameters: 110M'")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    imdb_dataset = load_dataset("imdb")
    print(imdb_dataset)

    # sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
    #
    # for row in sample:
    #     print(f"\n'>>> Review: {row['text']}'")
    #     print(f"'>>> Label: {row['label']}'")

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = imdb_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )
    print(tokenized_datasets)

