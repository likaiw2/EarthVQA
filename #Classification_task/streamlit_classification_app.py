import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import joblib
import numpy as np
import os
import tempfile
import sys

# 设置页面
st.set_page_config(page_title="Image Classification: CNN + XGBoost", layout="wide")
st.title("🏙️ Remote Sensing Image Classification (CNN + XGBoost)")

# 选择上传图片
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)

    # 选择 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)

    # --------------- CNN分类器预测 ----------------
    st.subheader("🔵 CNN Classifier Result")
    cnn_model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    cnn_model.fc = torch.nn.Linear(2048, 2)  # 输出2类
    cnn_model = cnn_model.to(device)

    # 加载CNN模型权重
    cnn_model.load_state_dict(torch.load('path/to/cnn_model.pth', map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, pred_cnn = torch.max(outputs, 1)
        label_cnn = int(pred_cnn.item())
    st.success(f"**CNN预测结果：{'Urban' if label_cnn == 1 else 'Rural'}**")

    # --------------- CNN+XGBoost分类器预测 ----------------
    st.subheader("🟠 CNN Features + XGBoost Classifier Result")
    
    # 特征提取
    feature_extractor = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    feature_extractor.fc = torch.nn.Identity()  # 去掉最后一层FC
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    with torch.no_grad():
        features = feature_extractor(img_tensor).cpu().numpy()

    # 加载XGBoost模型
    xgb_model = joblib.load('path/to/xgb_model.joblib')

    # 用特征进行预测
    pred_xgb = xgb_model.predict(features)
    label_xgb = int(pred_xgb[0])
    st.success(f"**CNN+XGBoost预测结果：{'Urban' if label_xgb == 1 else 'Rural'}**")