#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import numpy as np
import torch
import time
import logging
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_data(data_path, lang_code):
    df = pd.read_csv(f'{data_path}{lang_code}_parallel_detoxification.csv')
    hi_test_df = pd.read_csv(f'{data_path}hi_parallel_detoxification.test.csv')
    return df, hi_test_df

def split_data(df, seed_val):
    df_train, df_temp = train_test_split(df, test_size=600, random_state=seed_val)
    df_dev, df_test = train_test_split(df_temp, test_size=500, random_state=seed_val)
    return df_train, df_dev, df_test

def save_data(df_train, df_dev, df_test, hi_test_df):
    df_train.to_csv("data/en_train.csv", index=False)
    df_dev.to_csv("data/en_dev.csv", index=False)
    df_test.to_csv("data/en_test.csv", index=False)
    hi_test_df.to_csv("data/hi_test.csv", index=False)

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate(df, column, lang_code, tokenizer, model):
    start_time = time.time()

    inputs = tokenizer(df[column].to_list(), return_tensors="pt", padding=True, truncation=True, max_length=30)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code], max_length=30)
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    df[column] = translated_texts

    time_taken_seconds = time.time() - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))
    logging.info(f"Processed {column} for {len(df)} rows in {time_taken_formatted}")

    return df

def process_translation(df_train, df_dev, train_columns, lang_code, tokenizer, model):
    for column in train_columns:
        df_train = translate(df_train, column, lang_code, tokenizer, model)
        df_dev = translate(df_dev, column, lang_code, tokenizer, model)
    
    return df_train, df_dev

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../multilingual-detoxic-datasets_private/', help='Path to the dataset')
    parser.add_argument('--seed', type=int, default=53, help='Random seed value')
    parser.add_argument('--model_name', type=str, default="facebook/nllb-200-3.3B", help='Model name for translation')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_seed(args.seed)

    df, hi_test_df = load_data(args.data_path, "en")
    df_train, df_dev, df_test = split_data(df, args.seed)
    save_data(df_train, df_dev, df_test, hi_test_df)

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    train_columns = ['toxic_comment', 'civil_comment']
    hi_train_df, hi_dev_df = process_translation(df_train, df_dev, train_columns, 'hin_Deva', tokenizer, model)

    hi_train_df.to_csv("data/hi_train.csv", index=False)
    hi_dev_df.to_csv("data/hi_dev.csv", index=False)

if __name__ == '__main__':
    main()