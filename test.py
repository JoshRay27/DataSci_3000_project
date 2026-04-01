import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models.model_CNN import SimpleCNN
#from train import NUM_CLASSES   # Uses your existing constant

NUM_CLASSES = 2
MODEL_PATH = "simple_cnn_model.pth"
EVAL_DIR = "data_0_1"       # <-- folder containing class subfolders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_dataloader():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # FIX
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # adjust if RGB
    ])

    dataset = datasets.ImageFolder(EVAL_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset.classes

def evaluate(model, loader, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))

def main():
    print("Loading model...")
    model = load_model()

    print("Loading evaluation dataset...")
    loader, class_names = get_dataloader()

    print("Running evaluation...")
    evaluate(model, loader, class_names)

if __name__ == "__main__":
    main()