"""
train_rice_rf.py
-----------------
Trains a Random Forest classifier to detect rice leaf diseases from images.

HOW TO RUN THIS (in your Kaggle/Colab notebook, where the dataset is mounted):

    !pip install opencv-python-headless scikit-image scikit-learn joblib -q
    !python train_rice_rf.py

Or just copy this file's content into a notebook cell and run it directly.
Make sure `rice_features.py` is in the same folder / same working directory.

WHAT IT DOES:
  1. Walks the dataset folder (one sub-folder per disease class).
  2. Extracts a 51-value hand-crafted feature vector per image
     (color stats, color histograms, GLCM texture, edges, Hu moments,
     lesion-color ratio) using rice_features.py.
  3. Splits into train/test, trains a RandomForestClassifier.
  4. Prints accuracy, classification report, confusion matrix.
  5. Saves model.pkl, label_encoder.pkl, and a feature_importance.png
     so you (and the GUI) can reuse everything.

OUTPUT FILES (saved to OUTPUT_DIR):
  - rice_rf_model.joblib      <- the trained RandomForestClassifier
  - label_encoder.joblib      <- maps class index <-> disease name
  - feature_importance.png    <- bar chart of most useful features
  - confusion_matrix.png      <- visual model evaluation
"""

import os
import cv2
import glob
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from rice_features import extract_features

# ------------------------------------------------------------------
# CONFIG — edit these two paths for your environment
# ------------------------------------------------------------------
DATASET_DIR = "/kaggle/input/rice-disease-dataset/Rice_Leaf_AUG"  # <- change if needed
OUTPUT_DIR = "/kaggle/working"                                    # <- where to save model files
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# For quick experimentation you can cap the number of images loaded per
# class (the AUG dataset can be large). Set to None to use everything.
MAX_IMAGES_PER_CLASS = None   # e.g. set to 800 to speed up a first run


def load_dataset(dataset_dir):
    """Walk dataset_dir/<class_name>/*.jpg and extract features + labels."""
    class_names = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class sub-folders found in {dataset_dir}")

    print(f"Found {len(class_names)} classes: {class_names}")

    X, y = [], []
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths += glob.glob(os.path.join(class_dir, f"*{ext}"))
            paths += glob.glob(os.path.join(class_dir, f"*{ext.upper()}"))
        paths = sorted(set(paths))
        if MAX_IMAGES_PER_CLASS:
            paths = paths[:MAX_IMAGES_PER_CLASS]

        print(f"  {class_name}: {len(paths)} images")
        ok, bad = 0, 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                bad += 1
                continue
            try:
                feats = extract_features(img, is_bgr=True)
            except Exception:
                bad += 1
                continue
            X.append(feats)
            y.append(class_name)
            ok += 1
        if bad:
            print(f"    (skipped {bad} unreadable/corrupt files)")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset and extracting features (this can take a while)...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal samples: {X.shape[0]}, feature dimension: {X.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 5-fold cross-validation on the training set for a stability check
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    y_pred = clf.predict(X_test)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [f"feat_{i}" for i in order][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # Save model + label encoder
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "rice_rf_model.joblib"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

    print(f"\nSaved model, label encoder, and plots to: {OUTPUT_DIR}")
    print("Files:")
    for f in ["rice_rf_model.joblib", "label_encoder.joblib",
              "confusion_matrix.png", "feature_importance.png"]:
        print(" -", os.path.join(OUTPUT_DIR, f))


if __name__ == "__main__":
    main()
