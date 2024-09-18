from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from qa_dataset import QADataset


model = T5ForConditionalGeneration.from_pretrained('sberbank-ai/ruT5-base')
tokenizer = T5Tokenizer.from_pretrained('sberbank-ai/ruT5-base')

dataset = load_dataset("kuznetsoffandrey/sberquad")

train_data = dataset['train'].to_pandas()
valid_data = dataset['validation'].to_pandas()


def preprocess_data(df) -> tuple:
    """
    Preprocess a DataFrame to create input and target encodings for a model.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'context', 'question', and 'answers' columns.

    Returns:
        tuple: A tuple containing input encodings and target encodings.
    """
    inputs = []
    targets = []
    for _, row in df.iterrows():

        context = row['context']
        question = row['question']
        answer = row['answers']['text'][0]

        input_text = f"question: {question} context: {context}"
        inputs.append(input_text)
        targets.append(answer)

    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    target_encodings = tokenizer(targets, truncation=True, padding=True, max_length=128)

    return input_encodings, target_encodings


train_inputs, train_targets = preprocess_data(train_data)
valid_inputs, valid_targets = preprocess_data(valid_data)


train_dataset = QADataset(train_inputs, train_targets)
valid_dataset = QADataset(valid_inputs, valid_targets)
