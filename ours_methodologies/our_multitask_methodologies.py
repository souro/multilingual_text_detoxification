#!/usr/bin/env python
# coding: utf-8

import shutil
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from automatic_eval import TSTEvaluator
from argparse import ArgumentParser
import logging

# Set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Prepare dataframe for training
def prepare_df(pos, neg):
    pos = list(map(lambda s: s.strip(), pos))
    neg = list(map(lambda s: s.strip(), neg))
    data = {
        'src': pos + neg,
        'trg': neg + pos,
        'src_cls': [1] * len(pos) + [0] * len(neg),
        'trg_cls': [0] * len(neg) + [1] * len(pos)
    }
    return pd.DataFrame(data)

# Dataset class for input pairs
class CombinedDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.src = df['src']
        self.trg = df['trg']
        self.src_cls = df['src_cls']
        self.trg_cls = df['trg_cls']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_encoding = self.tokenizer(
            self.src[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt'
        )
        trg_encoding = self.tokenizer(
            self.trg[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': src_encoding.input_ids.squeeze(),
            'attention_mask': src_encoding.attention_mask.squeeze(),
            'labels': trg_encoding.input_ids.squeeze(),
            'src_cls': torch.tensor(self.src_cls[idx]),
            'trg_cls': torch.tensor(self.trg_cls[idx])
        }

# Custom Trainer with classifier heads and gradient reversal
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 2
        self.classifier_head = nn.Linear(1024, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def training_step(self, model, inputs):
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels = inputs["labels"].to(self.device)
        src_cls = inputs['src_cls'].to(self.device)
        trg_cls = inputs['trg_cls'].to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        seq2seq_loss = self.compute_loss(model, inputs)
        
        loss = seq2seq_loss
        
        # Encoder classification loss
        classification_logits_enc = self.classifier_head(outputs.encoder_last_hidden_state)
        one_hot_labels_enc = torch.nn.functional.one_hot(src_cls, self.num_classes)
        classification_loss_enc = self.criterion(classification_logits_enc, one_hot_labels_enc.float())
        loss += classification_loss_enc

        # Decoder classification loss
        classification_logits_dec = self.classifier_head(outputs.decoder_hidden_states[-1])
        one_hot_labels_dec = torch.nn.functional.one_hot(trg_cls, self.num_classes)
        classification_loss_dec = self.criterion(classification_logits_dec, one_hot_labels_dec.float())
        loss += classification_loss_dec

        loss.backward()
        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        ignore_keys = ['src_cls', 'trg_cls'] if ignore_keys is None else ignore_keys
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

# Generate predictions
def generate_predictions(test_df, model, tokenizer, src_column):
    def gen(src):
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=128, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=128)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return [gen(test_df[src_column].iloc[idx]) for idx in range(len(test_df))]

# Main function
def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', type=str, default='hi', help='Language code')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--methodology', type=str, default='seq2seqloss', help='Training methodology')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    lang = args.lang
    batch_size = args.batch_size
    methodology = args.methodology

    set_seed(53)

    train_df = pd.read_csv(f'data/{lang}_train.csv')
    dev_df = pd.read_csv(f'data/{lang}_dev.csv')
    test_df = pd.read_csv(f'data/{lang}_test.csv')

    train_df = prepare_df(train_df['toxic_comment'].tolist(), train_df['civil_comment'].tolist())
    dev_df = prepare_df(dev_df['toxic_comment'].tolist(), dev_df['civil_comment'].tolist())

    model_name = 'facebook/mbart-large-50'
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=f'{lang}_IN', tgt_lang=f'{lang}_IN')
    
    train_dataset = CombinedDataset(train_df, tokenizer)
    dev_dataset = CombinedDataset(dev_df, tokenizer)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    args = Seq2SeqTrainingArguments(
        output_dir=f"facebook-{methodology}-{lang}",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.evaluate()

    pred = generate_predictions(test_df, model, tokenizer, 'toxic_comment')
    
    output_file = f'output/{methodology}_{lang}.csv'
    output = {
        'src': test_df['toxic_comment'].tolist(),
        'trg': test_df['civil_comment'].tolist(),
        'pred': pred
    }
    pd.DataFrame(output).to_csv(output_file, index=False)

    # Evaluation
    tst_evaluator = TSTEvaluator(lang, output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()

    logging.info(f"Sentiment Accuracy: {accuracy}")
    logging.info(f"Similarity Score: {similarity_score}")
    logging.info(f"Bleu Score: {bleu}")
    logging.info(f"Fluency Score: {fluency_score}")

    # Sample outputs
    logging.info('-' * 10)
    for idx in range(10):
        logging.info(f"src: {test_df['toxic_comment'].iloc[idx]}")
        logging.info(f"trg: {test_df['civil_comment'].iloc[idx]}")
        logging.info(f"pred: {pred[idx]}\n")

if __name__ == '__main__':
    main()