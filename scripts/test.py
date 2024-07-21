from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

src_text = "Hello, how are you?"
inputs = tokenizer(src_text, return_tensors="pt")
translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id['fr_XX'])
tgt_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print("Translation:", tgt_text[0])
