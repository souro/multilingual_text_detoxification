#!/usr/bin/env python
# coding: utf-8

import shutil
import pandas as pd
import random
import numpy as np
import logging
from argparse import ArgumentParser
from automatic_eval import TSTEvaluator

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)

def load_data(lang):
    train_df = pd.read_csv(f'data/{lang}_train.csv')
    dev_df = pd.read_csv(f'data/{lang}_dev.csv')
    test_df = pd.read_csv(f'data/{lang}_test.csv')
    return train_df, dev_df, test_df

def load_vocab(lang):
    with open(f'toxic_vocab_{lang}.txt', 'r') as vocab_file:
        vocab_set = set(line.strip() for line in vocab_file)
    return vocab_set

def modify_text(test_df, src_column, vocab_set):
    modified_text_list = []
    for text in test_df[src_column]:
        words = text.split()
        filtered_words = [word for word in words if word not in vocab_set]
        modified_text = ' '.join(filtered_words)
        modified_text_list.append(modified_text)
    return modified_text_list

def save_output_file(test_df, modified_text_list, src_column, trg_column, output_file):
    output = {
        'src': test_df[src_column].values.tolist(),
        'trg': test_df[trg_column].values.tolist(),
        'pred': modified_text_list
    }
    output_df = pd.DataFrame(output)
    output_df.to_csv(output_file, index=False)

def display_samples(test_df, modified_text_list, src_column, trg_column, num_samples=10):
    logging.info('-' * 10)
    for idx in range(num_samples):
        logging.info(f'src: {test_df[src_column].values.tolist()[idx]}')
        logging.info(f'trg: {test_df[trg_column].values.tolist()[idx]}')
        logging.info(f'pred: {modified_text_list[idx]}')

def evaluate_output(lang, output_file):
    tst_evaluator = TSTEvaluator(lang, output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()
    
    logging.info(f"Sentiment Accuracy: {accuracy}")
    logging.info(f"Similarity Score: {similarity_score}")
    logging.info(f"Bleu Score: {bleu}")
    logging.info(f"Fluency Score: {fluency_score}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='hi', help='Language code (e.g., en, hi)')
    parser.add_argument('--src_column', type=str, default='toxic_comment', help='Source column in the dataset')
    parser.add_argument('--trg_column', type=str, default='civil_comment', help='Target column in the dataset')
    parser.add_argument('--seed', type=int, default=53, help='Random seed value')
    parser.add_argument('--output_file', type=str, default='output/delete_hi.csv', help='Output file path')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to display')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_seed(args.seed)

    train_df, dev_df, test_df = load_data(args.lang)
    vocab_set = load_vocab(args.lang)
    
    modified_text_list = modify_text(test_df, args.src_column, vocab_set)
    save_output_file(test_df, modified_text_list, args.src_column, args.trg_column, args.output_file)
    
    display_samples(test_df, modified_text_list, args.src_column, args.trg_column, args.num_samples)
    evaluate_output(args.lang, args.output_file)

if __name__ == '__main__':
    main()