import argparse
import logging
import torch
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math

import evaluate

class TSTEvaluator:
    def __init__(self, lang, output_file):
        self.lang = lang
        self.output_file = output_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def sentiment_accuracy(self, text, trg_label, lang):
        cls_model_name = "bert-base-multilingual-cased"
        cls_tokenizer = BertTokenizer.from_pretrained(cls_model_name)
        cls_model = BertForSequenceClassification.from_pretrained(cls_model_name,
                                                               num_labels=2,
                                                               output_attentions=False,
                                                               output_hidden_states=False)
        cls_model.to(self.device)
        cls_model.load_state_dict(torch.load(f'{lang}_cls_best.model'))

        inputs = cls_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = cls_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        predicted_labels = logits.argmax(dim=1).cpu().numpy()

        accuracy = accuracy_score(trg_label, predicted_labels)

        return accuracy

    def similarity(self, text1, text2, lang):
        sim_model = SentenceTransformer('sentence-transformers/LaBSE')

        sim_scores = list()
        for idx, _ in enumerate(text1):
            sentence_embedding1 = sim_model.encode(text1[idx])
            sentence_embedding2 = sim_model.encode(text2[idx])
            sim_score = cosine_similarity([sentence_embedding1], [sentence_embedding2])
            sim_scores.append(sim_score[0][0])

        return sum(sim_scores) / len(sim_scores)

    def bleu_score(self, pred, ref):
        bleu = evaluate.load("bleu")
        ref_bleu = list()
        for idx, text in enumerate(ref):
            ref_bleu.append([text])
        return bleu.compute(predictions=pred, references=ref_bleu, max_order=4)['bleu']

    def fluency(self, sentences):
        model_name = 'ai-forever/mGPT'  # 'sberbank-ai/mGPT'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()

        ppl = list()
        for sentence in sentences:
            tokenize_input = tokenizer.encode(sentence)
            tensor_input = torch.tensor([tokenize_input])

            with torch.no_grad():
                loss = model(tensor_input, labels=tensor_input)[0]
                # ppl.append(np.exp(loss.detach().numpy()))
                # print(np.exp(loss.detach().numpy()))
                if not math.isnan(np.exp(loss.detach().numpy())):
                    ppl.append(np.exp(loss.detach().numpy()))
        return sum(ppl) / len(ppl)

    def evaluate(self):
        output_df = pd.read_csv(self.output_file)
        output_df = output_df.fillna('not a valid string')
        target_label = 1

        accuracy = self.sentiment_accuracy(output_df['pred'].to_list(), [target_label] * len(output_df), self.lang)
        similarity_score = self.similarity(output_df['pred'].to_list(), output_df['trg'].to_list(), self.lang)
        bleu = self.bleu_score(output_df['pred'].to_list(), output_df['trg'].to_list())
        fluency_score = self.fluency(output_df['pred'].to_list())

        return accuracy, similarity_score, bleu, fluency_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TST Evaluator")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()

    logging.basicConfig(filename='tst_evaluator.log', level=logging.INFO)

    logging.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    tst_evaluator = TSTEvaluator(args.lang, args.output_file)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency_score = tst_evaluator.evaluate()

    logging.info(f'Sentiment Accuracy: {accuracy:.2f}')
    logging.info(f'Similarity: {similarity_score:.2f}')
    logging.info(f'Bleu Score: {bleu:.2f}')
    logging.info(f'Fluency: {fluency_score:.2f}')