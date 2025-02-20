import streamlit as st
import torch
import gdown
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import numpy as np
import matplotlib.pyplot as plt


# --- Google Drive File ID for Model ---
MODEL_FILE_ID = "1a2QvIB6kI6G79jUoSM2vxTkxQVZvERZU"
MODEL_FILE_PATH = "hybrid_model.pth"

# --- Download Model if not exists ---
@st.cache_resource
def download_model(file_id, output):
    model_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(model_url, output, quiet=False)
    st.success("Model downloaded successfully!")



# Custom Swin Transformer
class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super(CustomSwinTransformer, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.transformer_blocks = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, activation="gelu"),
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, activation="gelu")
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classification_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2).flatten(1)
        x = self.bottleneck(x)
        return self.classification_head(x)

# Hybrid Model combining Unet++ and Custom Swin Transformer
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.segmentation_model = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        self.classification_model = CustomSwinTransformer(num_classes=num_classes)

    def forward(self, x):
        seg_map = self.segmentation_model(x)
        seg_map = torch.sigmoid(seg_map)
        seg_map_resized = nn.functional.interpolate(seg_map, size=(224, 224), mode="bilinear", align_corners=False)
        combined = x * seg_map_resized
        return self.classification_model(combined)

# --- Load Model ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_model(MODEL_FILE_ID, MODEL_FILE_PATH)

    checkpoint = torch.load(MODEL_FILE_PATH, map_location=device)
    num_classes = len(checkpoint['class_labels'])
    model = HybridModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['class_labels'], device

model, class_labels, device = load_model()



# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, model, class_labels):
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        predicted_label = class_labels[predicted_class.item()]
    return predicted_label

def visualize_segmentation(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        seg_map = model.segmentation_model(image_tensor)
        seg_map = torch.sigmoid(seg_map)

    original_image = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    segmentation_map = seg_map.squeeze(0).squeeze(0).cpu().numpy()
    segmentation_map = (segmentation_map - segmentation_map.min()) / (segmentation_map.max() - segmentation_map.min())

    overlay = original_image.copy()
    overlay[:, :, 0] += segmentation_map

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(segmentation_map, cmap="jet")
    axs[1].set_title("Segmentation Map")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    st.pyplot(fig)

# Streamlit UI
st.title("Retinal Disease Detection with OCT scans")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    

    # Prediction
    label = predict(image, model, class_labels)
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100px; border: 2px solid #4CAF50; border-radius: 10px; background-color: rgba(0, 0, 0, 0); margin-top: 20px;">
            <h2 style="color: #4CAF50;">Predicted Class: {label}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Segmentation
    st.header("Segmentation Visualization")
    image_tensor = image_transform(image).unsqueeze(0)
    visualize_segmentation(model, image_tensor)
