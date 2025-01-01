from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer from checkpoint directory
tokenizer = AutoTokenizer.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")
model = AutoModelForSeq2SeqLM.from_pretrained("LeonardPuettmann/Quadrifoglio-mt-en-it")

def generate_response(input_text):
    model_text = f"translate English to Italian: {input_text}"
    input_ids = tokenizer(model_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Chatbot is ready. Type '/exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "/exit":
        print("Exiting the chatbot.")
        break
    response = generate_response(user_input)
    print("Chatbot:", response)