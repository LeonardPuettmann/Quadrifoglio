EN-IT on HuggingFace: https://huggingface.co/LeonardPuettmann/Quadrifoglio-mt-en-it
IT-EN on HuggingFace: https://huggingface.co/LeonardPuettmann/Quadrifoglio-mt-it-en

## 🍀 Quadrifoglio - A small model for English -> Italian translation

Quadrifoglio is an encoder-decoder transformer model for English-Italian text translation based on `bigscience/mt0-small`. It was trained on the `en-it` section of `Helsinki-NLP/opus-100` and `Helsinki-NLP/europarl`.


## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer from checkpoint directory
tokenizer = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")
model = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")

def generate_response(input_text):
    input_ids = tokenizer("translate English to Italian:" + input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

text_to_translate = "I would like a cup of green tea, please."
response = generate_response(text_to_translate)
print(response)
```

As this model is trained on translating sentence pairs, it is best to split longer text into individual sentences, ideally using SpaCy. You can then translate the sentences and join the translations at the end like this:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
# First, install spaCy and the English language model if you haven't already
# !pip install spacy
# !python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")
model = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")

def generate_response(input_text):
    input_ids = tokenizer("translate Italian to English: " + input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

text = "How are you doing? Today is a beautiful day. I hope you are doing fine."
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]

sentence_translations = []
for i, sentence in enumerate(sentences):
    sentence_translation = generate_response(sentence)
    sentence_translations.append(sentence_translation)

full_translation = " ".join(sentence_translations)
print(full_translation)
```

## Evaluation
Done on the Opus 100 test set.

### BLEU
|              | Quadrifoglio (this model)     | mt0-small| DeepL  |
|--------------|-------------------------------|----------|--------|
| BLEU Score   | 0.4816                        | 0.0159   | 0.5210 |
| Precision 1  | 0.7305                        | 0.2350   | 0.7613 | 
| Precision 2  | 0.5413                        | 0.0290   | 0.5853 | 
| Precision 3  | 0.4289                        | 0.0076   | 0.4800 |
| Precision 4  | 0.3417                        | 0.0013   | 0.3971 |
