from data_processing import model, tokenizer, train_dataset, valid_dataset
from torch.utils.data import DataLoader
import torch


LEARNING_RATE = 5e-5
BATCH_SIZE = 8
EPOCHS = 5


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)


for epoch in range(EPOCHS):  # Training
    model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        avg_train_loss = train_loss / (batch_idx + 1)
        print(f"Epoch {epoch}; Batch {batch_idx + 1}/{len(train_loader)}; Train_loss: {avg_train_loss:.4f}")

    model.eval()
    valid_loss = 0

    with torch.no_grad():  # Validation
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    print(f"Validation Loss: {avg_valid_loss:.4f}")

    model.save_pretrained(f"QA_{epoch}")
    tokenizer.save_pretrained(f"QA_{epoch}")
