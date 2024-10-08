from transformers import T5ForConditionalGeneration, T5Tokenizer

path_to_your_model = ''
TOKEN = ''

users_control = {}
first = True

model = T5ForConditionalGeneration.from_pretrained(path_to_your_model, use_safetensors=True)
tokenizer = T5Tokenizer.from_pretrained(path_to_your_model)
