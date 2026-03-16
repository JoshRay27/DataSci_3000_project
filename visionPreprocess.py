import cv2
import numpy as np
import os

def preprocess_live(img, size=(128,128), training=False):
    H, W, _ = img.shape

    # --- Skin detection ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)

    lower2 = np.array([160, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5,5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        blank = np.zeros((1, size[0], size[1]), dtype=np.float32)
        return blank, None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Square crop
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, W)
    y2 = min(cy + side // 2, H)

    # Padding
    pad = int(0.25 * side)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, W)
    y2 = min(y2 + pad, H)

    hand_region = img[y1:y2, x1:x2]  # <-- debug crop

    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    digit = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    digit = digit.astype("float32") / 255.0

    return digit.reshape(1, size[0], size[1]), (x1, y1, x2-x1, y2-y1)
    
def preprocess_image(img, file_name, size=(128,128), ):
    """
    Apply standard ML preprocessing to a single image.
    Convert to grayscale,
    resize,
    gaussian blur to redue noise,
    Normalize pixel values to [0, 1]
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (5,5),0)
    normalized = denoised.astype("float32") / 255.0
    print(f"Processed: {file_name}")
    return normalized

# Process folder just used for testing
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

"""
Grayscale simplifies the data

Models require fixed input size

Gaussian Blur removes noise and helps the model focus on structure rather than pixel-level randomness

Normalize makes pixel values small which is better for neural networks

Have to convert back to 0-255 to save with OpenCV
"""
