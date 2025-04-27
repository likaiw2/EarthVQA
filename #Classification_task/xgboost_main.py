"""
Image Classification Pipeline: CNN feature extraction + XGBoost classifier

Usage:
    python image_classification_pipeline.py \
        --data_dir path/to/images \
        --labels_csv path/to/labels.csv \
        --output_dir path/to/output \
        [--step {extract,train,predict,all}] \
        [--predict_image path/to/image.jpg]
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
from dataset import LoveDADataset
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

import xgboost; print(xgboost.__version__)

# ---------------------- Feature Extraction ----------------------
# 在 extract_features 里，除了对 Train 提取特征，也对 Val 提取特征
def extract_features(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # 训练集
    train_ds = LoveDADataset(root_dir=args.data_dir, split="Train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)
    # 验证集
    val_ds = LoveDADataset(root_dir=args.data_dir, split="Val", transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    def _extract(loader):
        feats, labs = [], []
        with torch.no_grad():
            for imgs, lbls, _ in tqdm(loader):
                imgs = imgs.to(device)
                out = model(imgs).cpu().numpy()
                feats.append(out)
                labs.extend(lbls.numpy())
        return np.vstack(feats), np.array(labs)

    print("Extracting training features...")
    X_train, y_train = _extract(train_loader)
    print("Extracting validation features...")
    X_val,   y_val   = _extract(val_loader)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'X_val.npy'),   X_val)
    np.save(os.path.join(args.output_dir, 'y_val.npy'),   y_val)
    print("saved features to", args.output_dir)

# ---------------------- Train Classifier ----------------------
def train_classifier(args):
    X_train = np.load(os.path.join(args.output_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.output_dir, 'y_train.npy'))
    X_val   = np.load(os.path.join(args.output_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(args.output_dir, 'y_val.npy'))

    from xgboost import callback

    clf = XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,           # 500 is the best in 100,200,300,400,500
        max_depth=3,                # 3 is the best in 3,5,7
        learning_rate=0.1,          # 0.1 is the best in 0.01,0.05,0.1,0.2
        subsample=0.9,            # 0.9 is the best in 0.5,0.7,0.8,0.9
        eval_metric='logloss'
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    # 最后在验证集上评估
    y_pred = clf.predict(X_val)
    print("Val Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    joblib.dump(clf, os.path.join(args.output_dir, 'xgb_model.joblib'))
    print(f"Model saved to {args.output_dir}")

# ---------------------- Predict Single Image ----------------------
def predict_image(args):
    # Load model
    model_path = os.path.join(args.output_dir, 'xgb_model.joblib')
    clf = joblib.load(model_path)

    # Load feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
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

    # Extract feature
    with torch.no_grad():
        feat = model(img_t).cpu().numpy()

    # Predict
    pred = clf.predict(feat)
    label = int(pred[0])
    print(f"Predicted label: {label} ({'Urban' if label==1 else 'Rural'})")

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description='Image Classification Pipeline')
    parser.add_argument('--data_dir', type=str,
                        default='/home/liw324/code/Segment/EarthVQA/dataset/LoveDA')
    parser.add_argument('--output_dir', type=str,
                        default='#Classification_task/out')
    parser.add_argument('--step', type=str, choices=['extract','train','predict','all'], 
                        default='train')
    parser.add_argument('--predict_image', type=str, 
                        default=None)
    args = parser.parse_args()

    # Example of data augmentation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if args.step in ['extract','all']:
        print("Extracting features...")
        extract_features(args)
        print("Done")
    if args.step in ['train','all']:
        print("Training classifier...")
        train_classifier(args)
        print("Done")
    if args.step in ['predict']:
        print("Predicting image...")
        if not args.predict_image:
            raise ValueError("--predict_image required for predict step")
        predict_image(args)
        print("Done")

if __name__ == '__main__':
    main()
