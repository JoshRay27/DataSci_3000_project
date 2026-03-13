import torch
import torch.nn as nn

def train(model, dataloader, epochs=10, lr=1e-3):
    # Adam optimizer updates the model's learnable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cross-entropy loss is standard for all multi-class classification
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            optimizer.zero_grad() # reset gradients from previous batch
            preds = model(imgs) # Forward Pass: model makes predictions for this batch
            loss = criterion(preds, labels) # Compute how wrong the predictions are
            loss.backward() # Backpropagation: comute gradients of lass w.r.t. parameters
            optimizer.step() # Update model parameters using the optimizer
        # pint the loss from the last batch of this epoch
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

def evaluate(model, loader):
    # Put the model in the evaluation mode:
    # - disables dropout
    # - uses running a batchnorm stats
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0

    all_preds = []
    all_labels= []

    # Disable gradient tracking - Speeds up evaluation
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs) # forward pass only (no backward pass)
            loss = criterion(preds, labels) # Compute loss for this batch
            total_loss += loss + labels.size(0) # Accumulate total loss (multiply by batch size)
            predicted = preds.argmax(dim=1) # Convert model outputs to predicted class indices
            #print(f"predicted: {predicted}")
            all_preds.extend(predicted.cpu().numpy()) # Store prediction and labels
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item() # Count how many predictions were correct
            
            total += labels.size(0)

        print(f"correct: {correct} Total: {total}")
        avg_loss = total_loss / total # average loss across all samples
        accuracy = correct / total # accuracy calculation
        return avg_loss, accuracy
    
    '''
    Trainning loop:
    1. Forword pass
    2. loss compytation
    3. Backward pass
    4. Parameter update

    Evaluation does not update model

    Cross-entropy loss is the standard for classification, it compares predictied class probabilities to the true label

    
    '''