import os
import sys
import cv2
import smtplib
from email.message import EmailMessage
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🔥",
    layout="wide"
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from dataset_utils import get_transforms  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Fire", "Non_Fire"]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🔥 Control Panel")

    mode = st.radio(
        "Detection Mode",
        ["Image Detection", "Live Video Detection"]
    )

    st.markdown("---")
    st.subheader("Alert Settings")

    FIRE_THRESHOLD = st.slider(
        "Fire Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.80,
        step=0.01
    )

    CONSEC_FRAMES = st.slider(
        "Consecutive Frames for Alert",
        min_value=1,
        max_value=10,
        value=5
    )

    st.markdown("---")
    st.info("Alerts are sent via Email with snapshot.")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🌲 Forest Fire Detection & Alert System</h1>
    <p style='text-align:center; font-size:18px;'>
    AI-powered real-time forest fire detection using Deep Learning & Computer Vision
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL ----------------
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "best_model.pth"
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------------- ALERT SYSTEM ----------------
def send_email_alert(image_path, confidence):
    msg = EmailMessage()
    msg["Subject"] = "🔥 FOREST FIRE ALERT"
    msg["From"] = os.getenv("SMTP_EMAIL")
    msg["To"] = os.getenv("ALERT_RECEIVER")

    msg.set_content(
        f"Forest Fire detected!\nConfidence: {confidence:.2f}\nImmediate action required."
    )

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename="fire_snapshot.jpg"
        )

    with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
        server.starttls()
        server.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
        server.send_message(msg)


# ---------------- INFERENCE ----------------
def predict_pil_image(image):
    _, eval_transform = get_transforms(224)
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return CLASS_NAMES[idx.item()], float(conf.item()), probs.cpu().tolist()


# ================= IMAGE MODE =================
if mode == "Image Detection":
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload Forest Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("Prediction Result")

        if uploaded and st.button("🔍 Analyze Image"):
            label, conf, probs = predict_pil_image(image)

            if label == "Fire":
                st.error("🔥 FIRE DETECTED")
            else:
                st.success("✅ NO FIRE DETECTED")

            st.metric("Confidence", f"{conf:.2%}")
            st.progress(conf)

            st.markdown("**Class Probabilities**")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- {cls}: {p:.3f}")

# ================= VIDEO MODE =================
else:
    st.warning("Press **Stop** in terminal to end monitoring")

    status_box = st.empty()
    frame_box = st.empty()

    if st.button("▶ Start Live Monitoring"):
        cap = cv2.VideoCapture(0)
        fire_count = 0
        alert_sent = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            label, conf, _ = predict_pil_image(img)

            if label == "Fire" and conf >= FIRE_THRESHOLD:
                fire_count += 1
                status_box.warning("⚠ Fire Suspected")
            else:
                fire_count = 0
                status_box.success("✅ Area Safe")

            if fire_count >= CONSEC_FRAMES and not alert_sent:
                snapshot = BASE_DIR / "fire_snapshot.jpg"
                cv2.imwrite(str(snapshot), frame)
                send_email_alert(snapshot, conf)
                status_box.error("🔥 ALERT SENT — FIRE CONFIRMED")
                alert_sent = True

            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255) if label == "Fire" else (0, 255, 0),
                2
            )

            frame_box.image(frame, channels="BGR")

        cap.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    ### 🔎 Status Legend
    - ✅ **Green** — No Fire Detected  
    - ⚠ **Yellow** — Possible Fire (Monitoring)  
    - 🔥 **Red** — Fire Confirmed & Alert Sent  
    """
)
