#!/usr/bin/env python
# coding: utf-8

import random
import time
import numpy as np
import pandas as pd
import torch
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_data(lang, src_column, trg_column):
    train_df = pd.read_csv(f'data/{lang}_train.csv')
    dev_df = pd.read_csv(f'data/{lang}_dev.csv')
    test_df = pd.read_csv(f'data/{lang}_test.csv')
    
    return train_df, dev_df, test_df

def prepare_data(df, src_column, trg_column):
    pos_data = df[trg_column].to_list()
    pos_labels = [1] * len(pos_data)
    neg_data = df[src_column].to_list()
    neg_labels = [0] * len(neg_data)
    
    combined_df = pd.DataFrame(list(zip(pos_data + neg_data, pos_labels + neg_labels)), columns=['Text', 'Label'])
    return combined_df.sample(frac=1)

def tokenize_data(tokenizer, text_data):
    return tokenizer.batch_encode_plus(
        text_data, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        max_length=128, 
        truncation=True, 
        padding='max_length', 
        return_tensors='pt'
    )

def create_dataloaders(train_data, dev_data, test_data, batch_size):
    train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], torch.tensor(train_data['Label'].values))
    dev_dataset = TensorDataset(dev_data['input_ids'], dev_data['attention_mask'], torch.tensor(dev_data['Label'].values))
    test_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'], torch.tensor(test_data['Label'].values))

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    return train_loader, dev_loader, test_loader

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions, true_vals = [], []

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            predictions.append(logits.detach().cpu().numpy())
            true_vals.append(inputs['labels'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return avg_loss, predictions, true_vals

def compute_f1_score(predictions, true_vals):
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def log_metrics(predictions, true_vals):
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()
    
    logging.info('Confusion Matrix')
    logging.info(confusion_matrix(labels_flat, preds_flat, labels=[1, 0]))
    logging.info('\nClassification Report\n')
    logging.info(classification_report(labels_flat, preds_flat))

def train_model(model, train_loader, dev_loader, optimizer, scheduler, epochs, device):
    best_val_loss = float('inf')
    early_stop_cnt = 0

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        start_time = time.time()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            model.zero_grad()

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, predictions, true_vals = evaluate(model, dev_loader, device)
        val_f1 = compute_f1_score(predictions, true_vals)

        logging.info(f'Epoch {epoch} | Train Loss: {avg_train_loss:.3f} | Val Loss: {val_loss:.3f} | F1 Score: {val_f1:.3f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 5:
            logging.info('Early stopping...')
            break

def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='en', help='Language code')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--seed', type=int, default=53, help='Random seed')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_seed(args.seed)

    train_df, dev_df, test_df = load_data(args.lang, 'toxic_comment', 'civil_comment')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    
    train_data = prepare_data(train_df, 'toxic_comment', 'civil_comment')
    dev_data = prepare_data(dev_df, 'toxic_comment', 'civil_comment')
    test_data = prepare_data(test_df, 'toxic_comment', 'civil_comment')

    train_enc = tokenize_data(tokenizer, train_data['Text'].values)
    dev_enc = tokenize_data(tokenizer, dev_data['Text'].values)
    test_enc = tokenize_data(tokenizer, test_data['Text'].values)

    train_loader, dev_loader, test_loader = create_dataloaders(train_enc, dev_enc, test_enc, args.batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_model(model, train_loader, dev_loader, optimizer, scheduler, args.epochs, device)

    model.load_state_dict(torch.load('best_model.pth'))
    _, predictions, true_vals = evaluate(model, test_loader, device)

    log_metrics(predictions, true_vals)

if __name__ == '__main__':
    main()