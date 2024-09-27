#!/usr/bin/env python
# coding: utf-8

import shutil
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, MBartForConditionalGeneration, MBart50TokenizerFast
from automatic_eval import TSTEvaluator
from argparse import ArgumentParser
import logging

# Set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Dataset class for input pairs
class CreateDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

# Generate predictions for test data
def generate_predictions(test_df, model, tokenizer, src_column, max_length=128):
    def gen(src):
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=max_length)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return [gen(test_df[src_column].values.tolist()[idx]) for idx in range(len(test_df))]

# Main function
def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='hi', help='Language code (e.g., en, hi)')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='kd_model/mbart-large-50/checkpoint-536', help='Model name or path')
    parser.add_argument('--src_column', type=str, default='toxic_comment', help='Source column name')
    parser.add_argument('--trg_column', type=str, default='civil_comment', help='Target column name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    lang = args.lang
    batch_size = args.batch_size
    set_seed(53)

    src_column = args.src_column
    trg_column = args.trg_column

    # Load data
    train_df = pd.read_csv(f'data/{lang}_train.csv')
    dev_df = pd.read_csv(f'data/{lang}_dev.csv')
    test_df = pd.read_csv(f'data/{lang}_test.csv')

    # Load tokenizer and model
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name, src_lang=f'{lang}_IN', tgt_lang=f'{lang}_IN')
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    # Tokenize data
    train_src_encodings = tokenizer(train_df[src_column].values.tolist(), truncation=True, padding=True, max_length=128)
    train_trg_encodings = tokenizer(train_df[trg_column].values.tolist(), truncation=True, padding=True, max_length=128)
    dev_src_encodings = tokenizer(dev_df[src_column].values.tolist(), truncation=True, padding=True, max_length=128)
    dev_trg_encodings = tokenizer(dev_df[trg_column].values.tolist(), truncation=True, padding=True, max_length=128)

    # Create datasets
    train_dataset = CreateDataset(train_src_encodings, train_trg_encodings)
    dev_dataset = CreateDataset(dev_src_encodings, dev_trg_encodings)

    # Set training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=args.model_name,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy='epoch',
        load_best_model_at_end=True,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train and evaluate the model
    trainer.train()
    trainer.evaluate()

    # Generate predictions
    pred = generate_predictions(test_df, model, tokenizer, src_column)

    # Save output to CSV
    output_file = f'output/knowledge_transfer_{lang}.csv'
    output = {
        'src': test_df[src_column].values.tolist(),
        'trg': test_df[trg_column].values.tolist(),
        'pred': pred
    }
    pd.DataFrame(output).to_csv(output_file, index=False)

    # Display sample results
    logging.info('-' * 10)
    for idx in range(10):
        logging.info(f'src: {test_df[src_column].values.tolist()[idx]}')
        logging.info(f'trg: {test_df[trg_column].values.tolist()[idx]}')
        logging.info(f'pred: {pred[idx]}\n')

    # Evaluate using TSTEvaluator
    tst_evaluator = TSTEvaluator(lang, output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()

    logging.info(f"Sentiment Accuracy: {accuracy}")
    logging.info(f"Similarity Score: {similarity_score}")
    logging.info(f"Bleu Score: {bleu}")
    logging.info(f"Fluency Score: {fluency_score}")

if __name__ == '__main__':
    main()