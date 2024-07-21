from datasets import load_dataset, concatenate_datasets
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
import torch

# Define model and tokenizer
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
model_name_fr_en = "Helsinki-NLP/opus-mt-fr-en"
print("Loading tokenizers and models...")
tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)
model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en)
print("Tokenizers and models loaded.")

# Load datasets
data_files = {
    "train_fr_eng": "./data/medical_jargon_translations_fr_eng.csv",
    "train_eng_fr": "./data/medical_jargon_translations_eng_fr.csv"
}
print("Loading datasets...")
datasets = load_dataset("csv", data_files=data_files)
print("Datasets loaded.")

# Preprocess functions
def preprocess_fr_eng(examples):
    inputs = examples['French']
    targets = examples['English']
    model_inputs = tokenizer_fr_en(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer_en_fr(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"].squeeze().tolist()
    return model_inputs

def preprocess_eng_fr(examples):
    inputs = examples['English']
    targets = examples['French']
    model_inputs = tokenizer_en_fr(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer_fr_en(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"].squeeze().tolist()
    return model_inputs

print("Tokenizing datasets...")
tokenized_fr_eng = datasets["train_fr_eng"].map(preprocess_fr_eng, batched=True)
tokenized_eng_fr = datasets["train_eng_fr"].map(preprocess_eng_fr, batched=True)
print("Datasets tokenized.")

# Combine datasets
combined_dataset = concatenate_datasets([tokenized_fr_eng, tokenized_eng_fr])
print("Datasets combined.")

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,  # Reduce batch size
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    fp16=True,  # Enable mixed precision training
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model_en_fr,  # Use the English-to-French model
    args=training_args,
    train_dataset=combined_dataset,
)
print("Trainer initialized.")

print("Starting training...")
# Check if CUDA is available and move the model to the GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    model_en_fr.to(device)
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

trainer.train()
print("Training completed.")

model_en_fr.save_pretrained("./model")
tokenizer_en_fr.save_pretrained("./model")
print("Model saved.")
