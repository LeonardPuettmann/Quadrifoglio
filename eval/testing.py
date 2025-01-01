from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm.auto import tqdm
import torch
import argparse
import requests
import json

def translate_deepl(auth_key, text, target_lang):
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "Content-Type": "application/json"
    }
    data = {
        "text": [text],
        "target_lang": target_lang
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()["translations"][0]["text"]
    else:
        print(f"Failed to translate text. Status code: {response.status_code}")
        print(response.json())
        return "None"

# Define the command line arguments
parser = argparse.ArgumentParser(description='Translation Evaluation Script')
parser.add_argument('-d', '--direction', type=str, default='en-it', choices=['en-it', 'it-en'], help='Translation direction (en-it or it-en)')
parser.add_argument('-m', '--model', type=str, default="LeonardPuettmann/Quadrifoglio-mt-it-en", help='Model to use for translation')
parser.add_argument('--deepl', action='store_true', help='Use DeepL API for translation')
parser.add_argument('--deepl_auth_key', type=str, help='DeepL authentication key')

args = parser.parse_args()

# Load model and tokenizer from checkpoint directory if not using DeepL
if not args.deepl:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # Move the model to the specified device
    model.eval()

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def preprocess_text(lines):
    translations = []
    lines = [line.strip() for line in lines if line.strip()]
    for i in range(0, len(lines), 3):
        english = lines[i]
        italian1 = lines[i+1]
        italian2 = lines[i+2]
        translations.append((english, italian1, italian2))
    return translations

lines = load_file(r"C:\Users\leopu\OneDrive\Programming\Python\llamaestro\eval\test-files\opus-2019-12-04.test.txt")
translations = preprocess_text(lines)

bleu = evaluate.load("bleu")

def translate(text):
    if args.deepl:
        if args.direction == 'en-it':
            target_lang = "IT"
        elif args.direction == 'it-en':
            target_lang = "EN"
        return translate_deepl(args.deepl_auth_key, text, target_lang)
    else:
        if args.direction == 'en-it':
            input_text = f"translate English to Italian: {text}"
        elif args.direction == 'it-en':
            input_text = f"translate Italian to English: {text}"
        input_ids = tokenizer(input_text, truncation=True, return_tensors="pt")["input_ids"].to(device)
        outputs = model.generate(input_ids, max_new_tokens=256)[0]
        return tokenizer.decode(outputs, skip_special_tokens=True)

predictions = []
references = []
sources = []
for translation in tqdm(translations):
    if args.direction == 'en-it':
        english = translation[0]
        italian_reference = translation[1]
        generated_italian = translate(english)
        predictions.append(generated_italian)
        references.append([italian_reference]) # BLEU score expects a list of references
        sources.append(english) # COMET metric expects the source text
    elif args.direction == 'it-en':
        italian = translation[1]
        english_reference = translation[0]
        generated_english = translate(italian)
        predictions.append(generated_english)
        references.append([english_reference]) # BLEU score expects a list of references
        sources.append(italian) # COMET metric expects the source text

# Calculate metrics
bleu_results = bleu.compute(predictions=predictions, references=references)

# Print metrics
print("BLEU:")
print(bleu_results)
