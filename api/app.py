import torch
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the models and tokenizers
model_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

model_fr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
tokenizer_fr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

def translate(text, model, tokenizer):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    with torch.no_grad():
        translated_tokens = model.generate(**tokens)
    
    # Decode the translated tokens
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get("text")
    direction = data.get("direction")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if direction == "en-fr":
        translated_text = translate(text, model_en_fr, tokenizer_en_fr)
    elif direction == "fr-en":
        translated_text = translate(text, model_fr_en, tokenizer_fr_en)
    else:
        return jsonify({"error": "Invalid translation direction"}), 400
    
    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
