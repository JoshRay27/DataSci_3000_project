import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class PreprocessedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = torch.tensor(np.array(img), dtype=torch.float32)
        img = img.unsqueeze(0)  # (1, H, W)
        return img, label