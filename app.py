import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model is in the same folder as app.py
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plant_disease_cnn_final.pth"
)

DEFAULT_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


# ============================================================
# CNN MODEL
# Must match the architecture used during training
# ============================================================

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


# ============================================================
# Helper
# ============================================================

def clean_class_name(name):

    return (
        name
        .replace("Tomato___", "")
        .replace("_", " ")
        .title()
    )


# ============================================================
# Load trained model
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "plant_disease_cnn_final.pth was not found. "
            "Keep the model file in the same folder as app.py."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    saved_classes = checkpoint.get(
        "classes",
        DEFAULT_CLASSES
    )

    model = CNN(
        num_classes=len(saved_classes)
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, saved_classes


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Uploaded image validation
# ============================================================

def validate_image(image):

    img = np.array(image)

    # Convert RGB → HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    # Green region
    lower_green = np.array(
        [25, 35, 25],
        dtype=np.uint8
    )

    upper_green = np.array(
        [95, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Remove small noise
    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find green regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = img.shape[0] * img.shape[1]

    significant_regions = [

        cv2.contourArea(contour)

        for contour in contours

        if cv2.contourArea(contour)
        > image_area * 0.01
    ]

    green_ratio = np.mean(mask > 0)

    # Lightweight heuristic
    if (
        len(significant_regions) > 4
        or green_ratio > 0.65
    ):

        return False

    return True


# ============================================================
# Prediction
# ============================================================

def predict_image(image, model):

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

    predicted_class = int(
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    confidence = float(
        probabilities[
            0,
            predicted_class
        ].item() * 100
    )

    return (
        input_tensor,
        predicted_class,
        confidence
    )


# ============================================================
# Grad-CAM
# ============================================================

def generate_gradcam(
    input_tensor,
    predicted_class,
    model
):

    # Last convolutional layer
    target_layer = model.features[9]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                predicted_class
            )
        ]
    )[0]

    return grayscale_cam


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Explainable Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


st.title(
    "🌿 Explainable Plant Disease Detection"
)

st.write(
    "Upload a clear photo of one tomato leaf. "
    "The CNN predicts the disease and Grad-CAM "
    "visualizes the regions that influenced the prediction."
)

st.info(
    "Supported classes: Bacterial Spot, "
    "Early Blight, Late Blight, and Healthy."
)


# ============================================================
# Load model
# ============================================================

try:

    model, classes = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# Upload image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # Read uploaded image
    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read the image. "
            "Please upload JPG, JPEG, or PNG."
        )

        st.stop()


    # Show uploaded image
    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="User uploaded image",
        use_container_width=True
    )


    # ========================================================
    # Image validation
    # ========================================================

    if not validate_image(image):

        st.error(
            "❌ Image not suitable for reliable prediction."
        )

        st.warning(
            "📸 Please upload a clear photo of ONE "
            "tomato leaf."
        )

        st.info(
            "Avoid hands, multiple leaves, "
            "and busy backgrounds."
        )

        st.stop()


    st.success(
        "✅ Image accepted. Proceeding to prediction."
    )


    # ========================================================
    # Prediction
    # ========================================================

    try:

        (
            input_tensor,
            predicted_class,
            confidence
        ) = predict_image(
            image,
            model
        )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.code(str(error))

        st.stop()


    disease = clean_class_name(
        classes[predicted_class]
    )


    # ========================================================
    # Result
    # ========================================================

    st.subheader(
        "🌿 Prediction Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Disease",
            disease
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    if confidence < 60:

        st.warning(
            "⚠️ Low confidence. "
            "Please upload a clearer single-leaf image."
        )

    else:

        st.success(
            "✅ Prediction completed."
        )


    # ========================================================
    # Grad-CAM explanation
    # ========================================================

    st.subheader(
        "🔍 Model Explanation — Grad-CAM"
    )

    try:

        grayscale_cam = generate_gradcam(
            input_tensor,
            predicted_class,
            model
        )

        # Resize original image
        display_image = np.array(
            image.resize((128, 128))
        ).astype(
            np.float32
        ) / 255.0

        # Generate visualization
        visualization = show_cam_on_image(
            display_image,
            grayscale_cam,
            use_rgb=True
        )

        st.image(
            visualization,
            caption=(
                "Grad-CAM: regions influencing "
                "the CNN prediction"
            ),
            use_container_width=True
        )

        st.caption(
            "Red/yellow regions indicate stronger "
            "contribution to the selected prediction. "
            "Grad-CAM is an importance visualization, "
            "not a precise disease-lesion map."
        )

    except Exception as error:

        st.warning(
            "Grad-CAM could not be generated."
        )

        st.code(str(error))


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Model trained on a four-class PlantVillage "
    "tomato-leaf subset. Real-world performance "
    "may differ because of domain shift."
)
