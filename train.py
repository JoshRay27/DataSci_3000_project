from torch.utils.data import DataLoader, random_split

from dataset import PreprocessedImageDataset
from model import SimpleCNN
from training import train, evaluate

DATA_DIR = "data/"
BATCH_SIZE = 3
NUM_CLASSES = 10

def main():
    # load full dataset
    dataset = PreprocessedImageDataset(DATA_DIR)

    # comput split sizes
    total = len(dataset)
    print(f"Length of dataset: {total}")

    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    #Perform Split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model_CNN = SimpleCNN(num_classes=NUM_CLASSES)

    train(model_CNN, train_loader, epochs=5, lr=1e-3)

    print("Model_CNN Evaluation")
    print(evaluate(model_CNN, val_loader))

if __name__ == "__main__":
    main()