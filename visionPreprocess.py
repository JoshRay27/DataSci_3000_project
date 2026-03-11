import cv2
import numpy as np
import os

def preprocess_image(img, size=(128,128)):
    """Apply standard ML preprocessing to a single image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (5,5),0)
    normalized = denoised.astype("float32") / 255.0
    return normalized

def process_folder(input_folder, output_folder, size=(128,128)):
    """Process all images in a folder and save the results. """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg",".png", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            processed = preprocess_image(img, size)

            # convert back to 0-255 for saving
            save_img = (processed * 255).astype("uint8")
            cv2.imwrite(output_path, save_img)

            print(f"Processed: {filename}")


process_folder(
    input_folder="archive/American Sign Language Digits Dataset/0/Input Images - Sign 0",
    output_folder="test0",
    size=(128,128)
)