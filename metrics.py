from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import re


path_to_your_model = ''

model = T5ForConditionalGeneration.from_pretrained(path_to_your_model, use_safetensors=True)
tokenizer = T5Tokenizer.from_pretrained("model")

dataset = load_dataset("kuznetsoffandrey/sberquad")

valid_data = dataset['validation'].to_pandas()
test_data = dataset['test'].to_pandas()


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


def clean_text(text: str) -> str:
    """
    Clean the input text by removing punctuation and converting it to lowercase.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def compute_f1_no_normalization(prediction: str, truth: str) -> float:
    """
    Compute the F1 score without text normalization.

    Args:
        prediction (str): The predicted text.
        truth (str): The ground truth text.

    Returns:
        float: The F1 score between the prediction and the truth.
    """
    pred_tokens = clean_text(text=prediction).split()
    truth_tokens = clean_text(text=truth).split()
    common_tokens = set(pred_tokens) & set(truth_tokens)

    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, truth: str) -> int:
    """
    Compute the Exact Match (EM) score between the predicted text and the ground truth text.

    Args:
        prediction (str): The predicted text.
        truth (str): The ground truth text.

    Returns:
        int: 1 if the cleaned prediction matches the cleaned truth, otherwise 0.
    """
    return int(clean_text(text=prediction) == clean_text(text=truth))


# Evaluating the model on the test dataset
for _, row in test_data.iterrows():

    con = row['context']
    q = row['question']

    print(f"Context: {con}")
    print(f"question: {q}")
    print(f"Answer: {generate_answer(question=q, context=con)}\n")


print('~' * 30, "calculate metrics", '~' * 30)
predictions = []
ground_truths = []

f1_scores = []
em_scores = []

BATCH = 100

# Calculation of metrics on the validation dataset (since the test dataset lacks targets)
for i, (_, row) in enumerate(valid_data.iterrows()):
    con = row['context']
    q = row['question']
    ans = row['answers']['text'][0]

    result = generate_answer(question=q, context=con)

    # Calculating F1 and EM for the current example
    f1 = compute_f1_no_normalization(prediction=result, truth=ans)
    em = compute_exact_match(prediction=result, truth=ans)

    f1_scores.append(f1)
    em_scores.append(em)

    # Displaying metrics every 100 examples
    if (i + 1) % BATCH == 0:
        avg_f1 = sum(f1_scores[-BATCH:]) / BATCH
        avg_em = sum(em_scores[-BATCH:]) / BATCH

        print(f"Batch {i + 1}: F1 = {avg_f1:.4f}, EM = {avg_em:.4f}")

# Displaying average metrics across the entire dataset
average_f1 = sum(f1_scores) / len(f1_scores)
average_em = sum(em_scores) / len(em_scores)

print(f"Final: F1 = {average_f1:.4f}, EM = {average_em:.4f}")
