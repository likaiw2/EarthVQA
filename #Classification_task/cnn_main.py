"""
Image Classification Pipeline: CNN classifier training and prediction

Usage:
    python image_classification_pipeline.py \
        --data_dir path/to/images \
        --labels_csv path/to/labels.csv \
        --output_dir path/to/output \
        [--step {train,predict}] \
        [--predict_image path/to/image.jpg]
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import LoveDADataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------------------- Train CNN Classifier ----------------------
def train_classifier(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = LoveDADataset(root_dir=args.data_dir, split="Train", transform=transform)
    val_ds = LoveDADataset(root_dir=args.data_dir, split="Val", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    model.fc = nn.Linear(2048, 2)  # 2 classes: Rural, Urban
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'cnn_model.pth'))
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

# ---------------------- Predict Single Image ----------------------
def predict_image(args):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    model.fc = nn.Linear(2048, 2)
    model = model.to(device)
    model_path = os.path.join(args.output_dir, 'cnn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load image
    img = Image.open(args.predict_image).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    label = int(pred.item())
    print(f"Predicted label: {label} ({'Urban' if label==1 else 'Rural'})")

    # --------------- CNN特征图可视化 ----------------
    import matplotlib.pyplot as plt

    # Extract features from the penultimate layer (before fc)
    # To do this, we can create a new model that outputs features
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # all layers except last fc
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        features = feature_extractor(img_t)
    features = features.view(features.size(0), -1)  # flatten to (1, 2048)

    features_np = features.squeeze(0).cpu().numpy()  # (2048,)

    # To visualize, we can reshape to (32, 64) for a rough 2D image
    feature_map = features_np.reshape(32, 64)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(feature_map, aspect='auto', cmap='viridis')
    fig.colorbar(cax)
    ax.set_title('CNN Extracted Feature Map')
    ax.axis('off')

    plt.show()

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description='Image Classification Pipeline')
    parser.add_argument('--data_dir', type=str,
                        default='/home/liw324/code/Segment/EarthVQA/dataset/LoveDA')
    parser.add_argument('--output_dir', type=str,
                        default='#Classification_task/out')
    parser.add_argument('--step', type=str, choices=['train','predict'], 
                        default='train')
    parser.add_argument('--predict_image', type=str, 
                        default=None)
    args = parser.parse_args()

    if args.step == 'train':
        print("Training CNN classifier directly on images...")
        train_classifier(args)
        print("Done")
    elif args.step == 'predict':
        print("Predicting image...")
        if not args.predict_image:
            raise ValueError("--predict_image required for predict step")
        predict_image(args)
        print("Done")

if __name__ == '__main__':
    main()
