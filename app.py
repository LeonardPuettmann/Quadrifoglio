import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

nlp_en = spacy.load("en_core_web_sm")
nlp_it = spacy.load("it_core_news_sm")

# Load translation models and tokenizers
tokenizer_en_it = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")
model_en_it = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")

tokenizer_it_en = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-it-en")
model_it_en = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-it-en")

def generate_response_en_it(input_text):
    input_ids = tokenizer_en_it("translate English to Italian: " + input_text, return_tensors="pt").input_ids
    output = model_en_it.generate(input_ids, max_new_tokens=256)
    return tokenizer_en_it.decode(output[0], skip_special_tokens=True)

def generate_response_it_en(input_text):
    input_ids = tokenizer_it_en("translate Italian to English: " + input_text, return_tensors="pt").input_ids
    output = model_it_en.generate(input_ids, max_new_tokens=256)
    return tokenizer_it_en.decode(output[0], skip_special_tokens=True)

def translate_text(input_text, direction):
    if direction == "en-it":
        nlp = nlp_en
        generate_response = generate_response_en_it
    elif direction == "it-en":
        nlp = nlp_it
        generate_response = generate_response_it_en
    else:
        return "Invalid direction selected."

    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]

    sentence_translations = []
    for sentence in sentences:
        sentence_translation = generate_response(sentence)
        sentence_translations.append(sentence_translation)

    full_translation = " ".join(sentence_translations)
    return full_translation

# Create the Gradio interface
iface = gr.Interface(
    fn=translate_text,
    inputs=[gr.Textbox(lines=5, placeholder="Enter text to translate..."),
            gr.Dropdown(choices=["en-it", "it-en"], label="Translation Direction")],
    outputs=gr.Textbox(lines=5, label="Translation")
)

# Launch the interface
iface.launch()
