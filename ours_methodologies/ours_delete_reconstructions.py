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
def generate_predictions(test_df, model, tokenizer, test_src, max_length=128):
    def gen(src):
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=max_length)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return [gen(test_df[test_src].values.tolist()[idx]) for idx in range(len(test_df))]

# Main function
def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='en', help='Language code')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    lang = args.lang
    batch_size = args.batch_size
    set_seed(53)

    src = 'civil_comment_delete'
    trg = 'civil_comment'
    test_src = 'toxic_comment_delete'

    # Load data
    train_df = pd.read_csv(f'data/{lang}_train.csv')
    dev_df = pd.read_csv(f'data/{lang}_dev.csv')
    test_df = pd.read_csv(f'data/{lang}_test.csv')

    # Remove directory if it exists
    shutil.rmtree('facebook', ignore_errors=True)
    model_name = 'facebook/mbart-large-50'

    # Set language codes
    src_lang_code = 'en_XX' if lang == 'en' else 'hi_IN'
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang_code, tgt_lang=src_lang_code)

    # Tokenize the data
    train_src_encodings = tokenizer(train_df[src].values.tolist(), truncation=True, padding=True, max_length=128)
    train_trg_encodings = tokenizer(train_df[trg].values.tolist(), truncation=True, padding=True, max_length=128)

    dev_src_encodings = tokenizer(dev_df[src].values.tolist(), truncation=True, padding=True, max_length=128)
    dev_trg_encodings = tokenizer(dev_df[trg].values.tolist(), truncation=True, padding=True, max_length=128)

    # Create datasets
    train_dataset = CreateDataset(train_src_encodings, train_trg_encodings)
    dev_dataset = CreateDataset(dev_src_encodings, dev_trg_encodings)

    # Load model
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Set training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=model_name,
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

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train and evaluate model
    trainer.train()
    trainer.evaluate()

    # Generate predictions for test data
    pred = generate_predictions(test_df, model, tokenizer, test_src)

    # Save output to CSV
    output_file = f'output/delete_recons_{lang}.csv'
    output = {
        'src': test_df['toxic_comment'].values.tolist(),
        'trg': test_df[trg].values.tolist(),
        'pred': pred
    }
    pd.DataFrame(output).to_csv(output_file, index=False)

    # Display sample results
    logging.info('-' * 10)
    for idx in range(10):
        logging.info(f'src: {test_df["toxic_comment"].values.tolist()[idx]}')
        logging.info(f'noisy_src: {test_df[test_src].values.tolist()[idx]}')
        logging.info(f'trg: {test_df[trg].values.tolist()[idx]}')
        logging.info(f'pred: {pred[idx]}\n')

    # Evaluate the results using TSTEvaluator
    tst_evaluator = TSTEvaluator(lang, output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()

    logging.info(f"Sentiment Accuracy: {accuracy}")
    logging.info(f"Similarity Score: {similarity_score}")
    logging.info(f"Bleu Score: {bleu}")
    logging.info(f"Fluency Score: {fluency_score}")

if __name__ == '__main__':
    main()