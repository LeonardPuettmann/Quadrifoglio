import spaces
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

class ModelSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelSingleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_it = spacy.load("it_core_news_sm")

            # Load translation models and tokenizers
            self.tokenizer_en_it = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")
            self.model_en_it = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")

            self.tokenizer_it_en = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-it-en")
            self.model_it_en = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-it-en")

            self.initialized = True

model_singleton = ModelSingleton()

@spaces.GPU(duration=30)
def generate_response_en_it(input_text):
    input_ids = model_singleton.tokenizer_en_it("translate English to Italian: " + input_text, return_tensors="pt").input_ids
    output = model_singleton.model_en_it.generate(input_ids, max_new_tokens=256)
    return model_singleton.tokenizer_en_it.decode(output[0], skip_special_tokens=True)

@spaces.GPU(duration=30)
def generate_response_it_en(input_text):
    input_ids = model_singleton.tokenizer_it_en("translate Italian to English: " + input_text, return_tensors="pt").input_ids
    output = model_singleton.model_it_en.generate(input_ids, max_new_tokens=256)
    return model_singleton.tokenizer_it_en.decode(output[0], skip_special_tokens=True)

def translate_text(input_text, direction):
    if direction == "en-it":
        nlp = model_singleton.nlp_en
        generate_response = generate_response_en_it
    elif direction == "it-en":
        nlp = model_singleton.nlp_it
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
    inputs=[gr.Textbox(lines=5, placeholder="Enter text to translate...", label="Input Text"),
            gr.Dropdown(choices=["en-it", "it-en"], label="Translation Direction")],
    outputs=gr.Textbox(lines=5, label="Translation"),
    description="This space is running on ZERO GPU. Initilization might take a couple of seconds the first time. This spaces uses the Quadrifoglio models for it-en and en-it text translation tasks."
)

# Launch the interface
iface.launch()

