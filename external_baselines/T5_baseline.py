#!/usr/bin/env python
# coding: utf-8

import shutil
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import gc
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from automatic_eval import TSTEvaluator
from tqdm.auto import tqdm, trange

# Set the seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Custom Dataset Class
class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.x.items()}
        item['decoder_attention_mask'] = self.y['attention_mask'][idx]
        item['labels'] = self.y['input_ids'][idx]
        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n

# Data collator with padding
class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True)
        ybatch = self.tokenizer.pad({'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']}, padding=True)
        batch['labels'] = ybatch['input_ids']
        batch['decoder_attention_mask'] = ybatch['attention_mask']
        return {k: torch.tensor(v) for k, v in batch.items()}

# Memory cleanup
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# Evaluation function for model
def evaluate_model(model, test_dataloader):
    total_loss = 0
    count = 0
    for batch in test_dataloader:
        with torch.no_grad():
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            total_loss += len(batch) * loss.item()
            count += len(batch)
    return total_loss / count

# Training loop
def train_loop(model, train_dataloader, val_dataloader, max_epochs, max_steps, lr, gradient_accumulation_steps, cleanup_step, report_step, window):
    cleanup()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    ewm_loss = 0
    step = 0
    model.train()

    for epoch in trange(max_epochs):
        if step >= max_steps:
            break
        tq = tqdm(train_dataloader)
        for i, batch in enumerate(tq):
            try:
                batch['labels'][batch['labels'] == 0] = -100
                loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
                loss.backward()
            except Exception as e:
                print('Error on step', i, e)
                cleanup()
                continue

            if i % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step >= max_steps:
                    break

            if i % cleanup_step == 0:
                cleanup()

            w = 1 / min(i+1, window)
            ewm_loss = ewm_loss * (1 - w) + loss.item() * w
            tq.set_description(f'loss: {ewm_loss:.4f}')

            if (i % report_step == 0 or i == len(train_dataloader) - 1) and val_dataloader is not None:
                model.eval()
                eval_loss = evaluate_model(model, val_dataloader)
                model.train()
                print(f'Epoch {epoch}, Step {i}/{step}: Train Loss: {ewm_loss:.4f}, Val Loss: {eval_loss:.4f}')

# Model training function
def train_model(x, y, model_name, test_size, batch_size, **kwargs):
    model = MT5ForConditionalGeneration.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=42)

    train_dataset = PairsDataset(tokenizer(x_train), tokenizer(y_train))
    val_dataset = PairsDataset(tokenizer(x_val), tokenizer(y_val))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    train_loop(model, train_dataloader, val_dataloader, **kwargs)
    return model

# Paraphrasing function
def paraphrase(text, model, tokenizer, n=None, max_length='auto', temperature=0.0, beams=3):
    inputs = tokenizer(text, return_tensors='pt', padding=True)['input_ids'].to(model.device)
    if max_length == 'auto':
        max_length = int(inputs.shape[1] * 1.2) + 10
    result = model.generate(
        inputs, num_return_sequences=n or 1, do_sample=False, temperature=temperature, repetition_penalty=3.0,
        max_length=max_length, num_beams=beams
    )
    return [tokenizer.decode(r, skip_special_tokens=True) for r in result]

# Main function
def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='hi', help='Language code (e.g., en, hi)')
    parser.add_argument('--model_name', type=str, default='google/mt5-base', help='Pre-trained model name')
    parser.add_argument('--output_file', type=str, default='../output/T5_hi.csv', help='Output file path')
    parser.add_argument('--steps', type=int, default=300, help='Number of training steps')
    args = parser.parse_args()

    set_seed(53)
    lang = args.lang
    model_name = args.model_name

    train_df = pd.read_csv(f'../data/{lang}_train.csv')
    test_df = pd.read_csv(f'../data/{lang}_test.csv')
    toxic_inputs = test_df['toxic_comment'].tolist()

    print(f"Training on {len(train_df)} samples")
    model = train_model(train_df['toxic_comment'].tolist(), train_df['civil_comment'].tolist(), model_name=model_name, batch_size=16, max_epochs=1000, max_steps=args.steps)
    model.save_pretrained(f't5_base_train_{args.steps}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(f't5_base_train_{args.steps}').cuda()

    para_results = []
    batch_size = 1

    for i in tqdm(range(0, len(toxic_inputs), batch_size)):
        batch = toxic_inputs[i:i + batch_size]
        para_results.extend(paraphrase(batch, model, tokenizer))

    output = {'src': test_df['toxic_comment'].tolist(), 'trg': test_df['civil_comment'].tolist(), 'pred': para_results}
    output_df = pd.DataFrame(output)
    output_df['pred'] = output_df['pred'].str.replace('<extra_id_0> ', '', regex=False)
    output_df.to_csv(args.output_file, index=False)

    # Evaluation
    tst_evaluator = TSTEvaluator(lang, args.output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()

    print("Accuracy:", accuracy)
    print("Similarity Score:", similarity_score)
    print("Bleu Score:", bleu)
    print("Fluency Score:", fluency_score)

if __name__ == '__main__':
    main()