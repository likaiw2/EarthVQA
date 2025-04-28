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
import matplotlib.pyplot as plt

# 设置页面
st.set_page_config(page_title="Image Classification: CNN + XGBoost", layout="wide")
st.title("🏙️ Remote Sensing Image Classification (CNN + XGBoost)")

# 选择上传图片
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

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

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        # --------------- CNN分类器预测 ----------------
        st.subheader("🔵 CNN Classifier Result")
        cnn_model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        cnn_model.fc = torch.nn.Linear(2048, 2)  # 输出2类
        cnn_model = cnn_model.to(device)

        # 加载CNN模型权重
        cnn_model.load_state_dict(torch.load('#Classification_task/out/cnn_model.pth', map_location=device))
        cnn_model.eval()

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            _, pred_cnn = torch.max(outputs, 1)
            label_cnn = int(pred_cnn.item())
        st.success(f"**CNN Result: {'Urban' if label_cnn == 1 else 'Rural'}**")

        # --------------- CNN+XGBoost分类器预测 ----------------
        st.subheader("🟠 CNN Features + XGBoost Classifier Result")
        
        # 特征提取
        feature_extractor = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        feature_extractor.fc = torch.nn.Identity()  # 去掉最后一层FC
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        with torch.no_grad():
            features = feature_extractor(img_tensor).cpu().numpy()

        # Streamlit checkbox to select PCA usage for XGBoost
        use_pca_xgb = st.checkbox("Use PCA before XGBoost prediction", value=True)

        if use_pca_xgb:
            pca_loaded_xgb, xgb_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/xgb_model_pca.joblib')
            features_pca_xgb = pca_loaded_xgb.transform(features)
            pred_xgb = xgb_loaded.predict(features_pca_xgb)
        else:
            xgb_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/xgb_model.joblib')
            pred_xgb = xgb_loaded.predict(features)

        label_xgb = int(pred_xgb[0])
        st.success(f"**CNN+XGBoost output: {'Urban' if label_xgb == 1 else 'Rural'}**")
        
        # --------------- CNN+SVM分类器预测 ----------------
        st.subheader("🟢 CNN Features + SVM Classifier Result")

        # 特征提取和前面一样，已经有 `features` 了

        # Streamlit checkbox to select PCA usage
        use_pca = st.checkbox("Use PCA before SVM prediction", value=True)

        if use_pca:
            pca_loaded, svm_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/svm_model_pca.joblib')
            features_pca = pca_loaded.transform(features)
            pred_svm = svm_loaded.predict(features_pca)
        else:
            svm_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/svm_model.joblib')
            pred_svm = svm_loaded.predict(features)

        label_svm = int(pred_svm[0])
        st.success(f"**CNN+SVM output: {'Urban' if label_svm == 1 else 'Rural'}**")
        
        # --------------- CNN+Random Forest分类器预测 ----------------
        st.subheader("🟣 CNN Features + Random Forest Classifier Result")

        # Streamlit checkbox to select PCA usage for RF
        use_pca_rf = st.checkbox("Use PCA before RF prediction", value=True)

        if use_pca_rf:
            pca_loaded_rf, rf_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/rf_model_pca.joblib')
            features_pca_rf = pca_loaded_rf.transform(features)
            pred_rf = rf_loaded.predict(features_pca_rf)
        else:
            rf_loaded = joblib.load('/home/liw324/code/Segment/EarthVQA/#Classification_task/out/rf_model.joblib')
            pred_rf = rf_loaded.predict(features)

        label_rf = int(pred_rf[0])
        st.success(f"**CNN+Random Forest output: {'Urban' if label_rf == 1 else 'Rural'}**")

    # --------------- CNN特征图可视化 ----------------
    col21, col22, col23 = st.columns(3)
    with col22:
        st.subheader("🔵 CNN Feature Map Visualization")

        # features shape is (1, 2048)
        features_np = features.squeeze(0)  # (2048,)

        # To visualize, we can reshape to (32, 64) for a rough 2D image
        feature_map = features_np.reshape(32, 64)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(feature_map, aspect='auto', cmap='viridis')
        fig.colorbar(cax)
        ax.set_title('CNN Extracted Feature Map')
        ax.axis('off')

        st.pyplot(fig)