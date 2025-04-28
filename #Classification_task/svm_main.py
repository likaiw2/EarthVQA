import os
import argparse
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

def train_svm(args):
    X_train = np.load(os.path.join(args.output_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.output_dir, 'y_train.npy'))
    X_val   = np.load(os.path.join(args.output_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(args.output_dir, 'y_val.npy'))

    if getattr(args, 'use_pca', False):
        # Use PCA before SVM
        print("Applying PCA before SVM...")
        pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        # Save PCA explained variance ratio information
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)
        np.save(os.path.join(args.output_dir, 'pca_explained_variance_ratio.npy'), explained_variance)
        np.save(os.path.join(args.output_dir, 'pca_cumulative_explained_variance.npy'), cumulative_explained_variance)
        print(f"PCA reduced feature dimension: {X_train_pca.shape[1]}")
        print(f"Saved PCA explained variance ratio to {args.output_dir}")
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_val_pca)
    else:
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(args, 'use_pca', False):
        joblib.dump((pca, clf), os.path.join(args.output_dir, 'svm_model_pca.joblib'))
        print(f"SVM+PCA model saved to {args.output_dir}")
    else:
        joblib.dump(clf, os.path.join(args.output_dir, 'svm_model.joblib'))
        print(f"SVM model saved to {args.output_dir}")

def predict_svm(args, features):
    if getattr(args, 'use_pca', False):
        model_path = os.path.join(args.output_dir, 'svm_model_pca.joblib')
        pca_loaded, svm_loaded = joblib.load(model_path)
        features = pca_loaded.transform(features)
    else:
        model_path = os.path.join(args.output_dir, 'svm_model.joblib')
        svm_loaded = joblib.load(model_path)
    preds = svm_loaded.predict(features)
    return preds

def main():
    parser = argparse.ArgumentParser(description='Train SVM classifier based on CNN features')
    parser.add_argument('--output_dir', type=str, 
                        default='#Classification_task/out')
    parser.add_argument('--use_pca', action='store_true', help='Apply PCA before SVM if set',
                        default=None)
    args = parser.parse_args()

    train_svm(args)

    # Load validation features and labels for demonstration prediction
    X_val = np.load(os.path.join(args.output_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(args.output_dir, 'y_val.npy'))
    y_pred_val = predict_svm(args, X_val)
    print("Prediction phase (after training) validation accuracy:", accuracy_score(y_val, y_pred_val))

if __name__ == "__main__":
    main()