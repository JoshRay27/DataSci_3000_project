import cv2
import torch
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from visionPreprocess import preprocess_live

NUM_CLASSES = 10
MODEL_PATH = "cnn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Model
model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def main():

    cap = cv2.VideoCapture(
        "nvarguscamerasrc ! nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )

    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("Camera failed to open")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            #print("ret:", ret, "frame:", None if frame is None else frame.shape)

            if not ret:
                break

            H, W, _ = frame.shape

            # Define a centered 400×400 ROI
            roi_size = 400
            x1 = W//2 - roi_size//2
            y1 = H//2 - roi_size//2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            roi = frame[y1:y2, x1:x2]
            
            processed, bbox = preprocess_live(roi)
            #print("processed shape:", processed.shape, "dtype:", processed.dtype)
            #print("min/max:", processed.min(), processed.max())

            tensor = torch.from_numpy(processed).float().unsqueeze(0).to(device)
 
            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output, dim=1).item()

            cv2.putText(frame, f"Prediction: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# Draw bounding box if found
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(roi, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)


            cv2.imshow("Processed", processed[0])
            cv2.imshow("Camera", frame)
            print(f"Prediction: {pred}")
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

