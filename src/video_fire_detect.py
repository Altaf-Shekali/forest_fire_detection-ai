import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from dataset_utils import get_transforms  # uses same transforms as training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match training class order: ['Fire', 'Non_Fire']
CLASS_NAMES = ["Fire", "Non_Fire"]

# Fire detection logic
FIRE_PROB_THRESHOLD = 0.8        # minimum probability to consider "fire"
CONSEC_FRAMES_FOR_ALERT = 5      # how many frames in a row before alert


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model():
    base_dir = Path(__file__).resolve().parent.parent  # project root
    model_path = base_dir / "models" / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = build_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_frame(model, frame_bgr):
    """
    frame_bgr: numpy array from OpenCV (BGR)
    returns: (label, confidence, probs)
    """
    # Convert BGR -> RGB -> PIL Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Use eval transform (not train)
    _, eval_transform = get_transforms(image_size=224)
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        score, pred_idx = torch.max(probs, dim=0)

    label = CLASS_NAMES[pred_idx.item()]
    confidence = float(score.item())
    return label, confidence, probs.cpu().tolist()


def simple_alert():
    """
    Simple local alert. For now, just prints.
    You can later extend: send email / Telegram, sound buzzer, etc.
    """
    print("🔥🔥🔥 ALERT: FIRE DETECTED CONSISTENTLY IN VIDEO! 🔥🔥🔥")


def run_video_inference(source=0):
    """
    source:
      0            -> webcam
      'path.mp4'  -> video file path
    """
    model = load_model()
    print("Model loaded. Device:", DEVICE)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    consec_fire_frames = 0
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or cannot read from source.")
            break

        label, confidence, probs = predict_frame(model, frame)

        # Update fire frame counter
        if label == "Fire" and confidence >= FIRE_PROB_THRESHOLD:
            consec_fire_frames += 1
        else:
            consec_fire_frames = 0  # reset if not strong fire

        # Trigger alert once when threshold is crossed
        if consec_fire_frames >= CONSEC_FRAMES_FOR_ALERT and not alert_triggered:
            simple_alert()
            alert_triggered = True  # avoid spamming

        # Draw overlay on frame
        text = f"{label} ({confidence:.2f})"
        color = (0, 255, 0) if label == "Non_Fire" else (0, 0, 255)  # green / red

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        # Show alert status
        if alert_triggered:
            cv2.putText(
                frame,
                "ALERT TRIGGERED",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Forest Fire Video Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Usage:
      python src/video_fire_detect.py           -> webcam
      python src/video_fire_detect.py video.mp4 -> video file
    """
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        # If user passes "0", treat as webcam
        if source_arg == "0":
            source = 0
        else:
            source = source_arg  # assume file path
    else:
        source = 0  # default webcam

    run_video_inference(source)


if __name__ == "__main__":
    main()
