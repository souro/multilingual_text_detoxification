#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random
import torch
import logging
from argparse import ArgumentParser
from transformers import BertTokenizer, BertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_model_and_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False, output_hidden_states=False)
    return model, tokenizer

def merge_contractions(attributions):
    merged_attributions = []
    i = 0
    while i < len(attributions):
        word, score = attributions[i]
        if i + 2 < len(attributions) and attributions[i+1][0] == "'" and attributions[i+2][0] in ["t", "re", "ve", "s", "m", "ll", "d"]:
            merged_word = word + attributions[i+1][0] + attributions[i+2][0]
            merged_score = score + attributions[i+1][1] + attributions[i+2][1]
            merged_attributions.append((merged_word, merged_score))
            i += 3
        else:
            merged_attributions.append((word, score))
            i += 1
    return merged_attributions

def calculate_attributions(cls_explainer, sentences, lang='en', threshold=0.5):
    masked_sentences = []
    for sentence in sentences:
        attributions = cls_explainer(sentence)
        word_attributions = []
        word, word_score = "", 0.0
        for token, score in attributions:
            if token.startswith("##"):  # Handle subwords
                word += token[2:]
                word_score += score
            else:
                if word:
                    word_attributions.append((word, word_score))
                word = token
                word_score = score
        if word:
            word_attributions.append((word, word_score))

        merged_attributions = merge_contractions(word_attributions) if lang == 'en' else word_attributions

        masked_sentence = [word if score < threshold and word not in ['[CLS]', '[SEP]'] else '' for word, score in merged_attributions]
        masked_sentences.append(" ".join(masked_sentence))

    return masked_sentences

def mask_sentences(df, column_name, model, tokenizer, lang='en', threshold=0.5):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    sentences = df[column_name].tolist()
    masked_sentences = calculate_attributions(cls_explainer, sentences, lang=lang, threshold=threshold)
    df[f"{column_name}_delete"] = masked_sentences
    return df

def load_dataframes():
    return {
        'en_train': pd.read_csv('data/en_train.csv'),
        'en_dev': pd.read_csv('data/en_dev.csv'),
        'en_test': pd.read_csv('data/en_test.csv'),
        'hi_train': pd.read_csv('data/hi_train.csv'),
        'hi_dev': pd.read_csv('data/hi_dev.csv'),
        'hi_test': pd.read_csv('data/hi_test.csv')
    }

def save_dataframes(dataframes):
    for df_name, df in dataframes.items():
        df.to_csv(f'data/{df_name}.csv', index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased", help='Model name')
    parser.add_argument('--seed', type=int, default=53, help='Random seed value')
    parser.add_argument('--threshold', type=float, default=0.5, help='Attribution threshold')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataframes = load_dataframes()

    for df_name, df in dataframes.items():
        lang = 'hi' if 'hi' in df_name else 'en'
        for column_name in ['toxic_comment', 'civil_comment']:
            df = mask_sentences(df, column_name, model, tokenizer, lang=lang, threshold=args.threshold)

    save_dataframes(dataframes)

if __name__ == '__main__':
    main()