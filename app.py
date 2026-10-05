import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Classes
# -----------------------------
classes = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


# -----------------------------
# CNN Architecture
# -----------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():

    model = CNN(num_classes=4).to(device)

    checkpoint = torch.load(
        "/content/plant_disease_cnn_final.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model = load_model()


# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Image suitability check
# -----------------------------
def validate_image(image):

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 35, 25])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = img.shape[0] * img.shape[1]

    regions = [
        cv2.contourArea(c)
        for c in contours
        if cv2.contourArea(c) > image_area * 0.01
    ]

    green_ratio = np.sum(mask > 0) / image_area

    if len(regions) > 4 or green_ratio > 0.65:
        return False

    return True


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Explainable Plant Disease Detection")

st.write(
    "Upload a clear photo of one tomato leaf. "
    "The CNN predicts the disease and Grad-CAM "
    "shows the regions influencing the prediction."
)

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if not validate_image(image):

        st.error(
            "Image not suitable for reliable prediction. "
            "Please upload a clear photo of ONE tomato leaf."
        )

        st.info(
            "Avoid hands, multiple leaves, and busy backgrounds."
        )

    else:

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)

        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item() * 100

        disease = classes[predicted_class].replace(
            "Tomato___", ""
        )

        st.success(f"Prediction: {disease}")
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # Grad-CAM
        cam = GradCAM(
            model=model,
            target_layers=[model.features[9]]
        )

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[
                ClassifierOutputTarget(predicted_class)
            ]
        )[0]

        display_image = np.array(
            image.resize((128, 128))
        ).astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            display_image,
            grayscale_cam,
            use_rgb=True
        )

        st.subheader("🔍 Model Explanation — Grad-CAM")

        st.image(
            visualization,
            caption="Regions influencing the CNN prediction",
            use_container_width=True
        )

        if confidence < 60:
            st.warning(
                "Low confidence. Please upload a clearer "
                "single-leaf image."
            )
