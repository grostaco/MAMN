from __future__ import annotations


from transformers import (BertTokenizer,
                          BertModel,
                          Trainer,
                          TrainingArguments,
                          EvalPrediction)
from datasets import load_dataset, DatasetDict
from typing import cast
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report)

from lib.model import MAMNConfig, MAMN

bert = BertModel.from_pretrained('bert-base-uncased')
dataset = cast(DatasetDict, load_dataset('grostaco/laptops-trial'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label2id = dataset['train'].features['labels']._str2int
id2label = {v: k for k, v in label2id.items()}

config = MAMNConfig(label2id=label2id, id2label=id2label)
model = MAMN(config)


def tokenize_aspects(aspects):
    tokenized = tokenizer(aspects, padding='max_length',
                          truncation=True, max_length=16, return_tensors='pt')

    return {f'aspects_{k}': v for k, v in tokenized.items()}


def tokenize(contents):
    return tokenizer(contents, padding='max_length', truncation=True, max_length=256, return_tensors='pt')


dataset = dataset.map(tokenize, input_columns='content', batched=True)
dataset = dataset.map(tokenize_aspects, input_columns='aspect', batched=True)
dataset = dataset.filter(lambda end: end < 80, input_columns='end')


def compute_metrics(p: EvalPrediction):
    y_pred = p.predictions[0].argmax(-1)
    print(y_pred)
    y_true = p.label_ids

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    report = classification_report(
        y_true, y_pred, target_names=label2id.keys())

    print(report)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'recall': recall,
    }


args = TrainingArguments(
    output_dir='mamn',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    num_train_epochs=1,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

trainer.train()
