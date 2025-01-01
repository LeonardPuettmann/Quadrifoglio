#!/usr/bin/env python

import argparse
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import evaluate
import numpy as np
from torch.optim import Adafactor
from evaluate import load
from tqdm.auto import tqdm

wandb.login()
run = wandb.init(
    project='Fine-tune mt0 on Italian-English language translation data with opus', 
    job_type="training", 
    anonymous="allow"
)

# Define arguments
parser = argparse.ArgumentParser(
    prog="T5 trainer script",
    description="Trains T5 on a dataset",
    epilog="Use at your own risk!"
)
parser.add_argument('-m', '--model', help="The model checkpoint to train", type=str)
parser.add_argument('-d', '--dataset', help="The dataset to train on", type=str)
parser.add_argument('-e', '--epochs', help="The epoch number to append to the end of the filename", type=int)
parser.add_argument('-lr', '--learning_rate', help="The learning rate for training", type=float)

args = parser.parse_args()

# Set default values
model = args.model if args.model else "bigscience/mt0-small"
dataset = load_dataset(args.dataset) if args.dataset else load_dataset("Helsinki-NLP/opus-100", "en-it")
epochs = int(args.epochs) if args.epochs else 1
lr = float(args.learning_rate) if args.learning_rate else 2e-05

# Preprocess data
raw_dataset_train_test = dataset["train"].train_test_split(test_size=0.3, seed=42)
raw_dataset_test_val = raw_dataset_train_test["test"].train_test_split(test_size=0.01, seed=42)

# Initialize tokenizer
tokenizer  = AutoTokenizer.from_pretrained(model)
model = AutoModelForSeq2SeqLM.from_pretrained(model)

# Set seed
#set_seed(42)

# Define source and target languages
source_lang = "it"
target_lang = "en"
prefix = "translate Italian to English: "

# Preprocess function
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target=targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Define compute metrics function
def compute_metrics_after_run(predictions, references):
    results = load("bleu").compute(predictions=predictions, references=references)
    return results

# Tokenize data
tokenized_train = raw_dataset_train_test['train'].map(preprocess_function, batched=True)
tokenized_test = raw_dataset_test_val["test"].map(preprocess_function, batched=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define translation function
def translate(text):
    input_ids = tokenizer(f"translate Italian to English: {text}", return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(input_ids=input_ids)[0]
    return tokenizer.decode(outputs, skip_special_tokens=True)

# Define get label function
def get_label(example, lang):
    return {"references":example["translation"][lang]}

# %%
# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results-mt0-opus-it-en",
    eval_strategy="steps",
    eval_steps=5000,
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.0,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=False,
    bf16=True,
    push_to_hub=False,
    report_to="wandb"
)

optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),  # Pass custom optimizer
)

trainer.train()

# %%

model.save_pretrained(f"./mt0_finetuned_opus_it_en_{epochs}")
tokenizer.save_pretrained(f"./mt0_finetuned_opus_it_en_{epochs}")

torch.backends.cuda.matmul.allow_tf32 = True

model.eval()

# Evaluate model
it_labels = raw_dataset_test_val.map(get_label, fn_kwargs={'lang':'it'})
en_labels = raw_dataset_test_val.map(get_label, fn_kwargs={'lang':'en'})
predictions = []
it_references = []
en_references = []
for label in it_labels["test"]:
    it_references.append(label["references"])
for label in en_labels["test"]:
    en_references.append(label["references"])
for i in tqdm(range(len(raw_dataset_test_val["test"]["translation"]))): # raw_dataset_test_val["test"]
    predictions.append(translate(raw_dataset_test_val["test"]["translation"][i]["it"]))

# Compute metrics
print("BLEU:")
print(compute_metrics_after_run(predictions=predictions, references=en_references))

# %%
comet_metric = evaluate.load('comet')
source = it_references
comet_score = comet_metric.compute(predictions=predictions, references=en_references, sources=source)
print("Comet:")
print(comet_score['mean_score'])

# %%

bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=predictions, references=en_references, model_type="distilbert-base-uncased")
print("Bertscore:")

def get_average(l:list):
    return (sum(l)/len(l))

avg_precision = get_average(results["precision"])
avg_recall = get_average(results["recall"])
avg_f1 = get_average(results["f1"])
print({"average_precision":avg_precision, "average_recall":avg_recall, "average_f1":avg_f1})

# meteor = evaluate.load("meteor")
# results = meteor.compute(predictions=predictions, references=it_references)
# print("Meteor:")
# print(results)