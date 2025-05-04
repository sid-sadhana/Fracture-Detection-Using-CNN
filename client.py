import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO

st.title("CrackTrack")

uploaded_file = st.file_uploader("Upload an X-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    def load_model(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        conv1 = nn.Conv2d(1, 32, 3)
        conv2 = nn.Conv2d(32, 64, 3)
        conv3 = nn.Conv2d(64, 128, 3)
        fc1 = nn.Linear(128 * 26 * 26, 256)
        fc2 = nn.Linear(256, 128)
        fc3 = nn.Linear(128, 1)

        conv1.load_state_dict(checkpoint['conv1_state_dict'])
        conv2.load_state_dict(checkpoint['conv2_state_dict'])
        conv3.load_state_dict(checkpoint['conv3_state_dict'])
        fc1.load_state_dict(checkpoint['fc1_state_dict'])
        fc2.load_state_dict(checkpoint['fc2_state_dict'])
        fc3.load_state_dict(checkpoint['fc3_state_dict'])

        model = nn.Sequential(
            conv1, nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv2, nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv3, nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            fc1, nn.LeakyReLU(),
            fc2, nn.LeakyReLU(),
            fc3, nn.Sigmoid()
        )

        model.eval()
        return model

    def preprocess_image(image):
        image = transform(image).unsqueeze(0)
        return image

    def predict_fracture(image, model):
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.round(output).item()
            return "No Fracture" if prediction == 1 else "Fracture"

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    model_path = r'C:\Users\srisi\OneDrive\Desktop\sidxt\showcase\Fracture-Detection-Using-CNN\main.pth'
    model = load_model(model_path)

    result = predict_fracture(image, model)
    st.subheader(f"Prediction: {result}")
