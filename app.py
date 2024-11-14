import streamlit as st
from PIL import Image  
import torchvision.transforms as transforms 
import torch
from src.network import SiameseNetwork 
import torch.nn.functional as F

st.title("Face Dissimilarity Detection")

# Load model
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

# File uploader
uploaded_file1 = st.file_uploader("Choose first image...", type="jpg")
uploaded_file2 = st.file_uploader("Choose second image...", type="jpg")

if uploaded_file1 is not None and uploaded_file2 is not None:
    img1 = Image.open(uploaded_file1).convert("L")
    img2 = Image.open(uploaded_file2).convert("L")
    
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    
    output1, output2 = model(img1_tensor, img2_tensor)
    euclidean_distance = F.pairwise_distance(output1, output2)
    
    st.image([img1, img2], caption=["Image 1", "Image 2"])
    st.write(f"Dissimilarity Score: {euclidean_distance.item():.2f}")
