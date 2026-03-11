from torch.utils.data import DataLoader
from dataset import PreprocessedImageDataset

train_ds = PreprocessedImageDataset("dataset/")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)