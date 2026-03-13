import torch
import torch.nn as nn

def train(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0

    all_preds = []
    all_labels= []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs)
            loss = criterion(preds, labels)
            total_loss += loss + labels.size(0)
            predicted = preds.argmax(dim=1)
            #print(f"predicted: {predicted}")
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            
            total += labels.size(0)

        print(f"correct: {correct} Total: {total}")
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    