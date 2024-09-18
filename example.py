from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained("model", use_safetensors=True)
tokenizer = T5Tokenizer.from_pretrained("model")


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer based on the provided question and context.

    Args:
        question (str): The question to be answered.
        context (str): The context containing relevant information for generating the answer.

    Returns:
        str: The generated answer.
    """
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
