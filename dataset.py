import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from visionPreprocess import preprocess_image
import cv2

class PreprocessedImageDataset(Dataset):
    def __init__(self, root_dir):
        # This list will store tuples of (image_path, labels)
        self.samples = []
        for label, class_name in enumerate(sorted(os.listdir(root_dir))): # Loop through each subfolder inside rood_dir
            class_path = os.path.join(root_dir, class_name) # each subfolder name is treaded as a class
            if not os.path.isdir(class_path): # skip anything that is not a folder
                continue
            for fname in os.listdir(class_path): # loop through all files inside the class folder
                if fname.lower().endswith((".png", ".jpg", ".jpeg")): # only keep image files
                    self.samples.append((os.path.join(class_path, fname), label)) # store the full path and the numeric label

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the (image_path, label) pair at the given index
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path) # Load the image using OpenCV (BGR format)
        img = preprocess_image(img, file_name=img_path) # (resize, grayscale, normalization, etc)
        # add channel dimension: (H, W) -> (1, H, W)
        # PyTorch expects images in (C, H, W) format
        img = torch.tensor(np.array(img), dtype=torch.float32)
        img = img.unsqueeze(0)  # (1, H, W)

        # Return the image tensor and its label
        return img, label
    
    """
    Output its setup for pytorch
    """