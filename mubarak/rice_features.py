"""
rice_features.py
-----------------
Feature extraction for the Rice Leaf Disease Random Forest model.

Random Forest cannot learn from raw pixels the way a CNN does, so we
convert each leaf image into a compact vector of hand-crafted features
that describe color and texture, both of which are highly discriminative
for plant leaf diseases (lesion color, spot texture, blight patterns).

Feature groups (per image), ~63 features total:
  1. Color statistics (RGB + HSV mean/std)           -> 12 features
  2. Color histograms (HSV, 8 bins per channel)       -> 24 features
  3. GLCM texture (contrast, homogeneity, energy,
     correlation, dissimilitude, ASM)                 -> 6 features
  4. Edge density (Canny)                              -> 1 feature
  5. Hu moments (shape of dark lesion regions)         -> 7 features
  6. Lesion/spot ratio (fraction of "diseased-looking"
     pixels via HSV thresholding)                      -> 1 feature

Keeping this in its own module means the training script and the GUI
app use IDENTICAL feature extraction, which is essential for a Random
Forest model to work correctly at inference time.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

IMG_SIZE = 128  # all images resized to this before feature extraction


def _color_stats(img_rgb, img_hsv):
    feats = []
    for img in (img_rgb, img_hsv):
        for c in range(3):
            channel = img[:, :, c]
            feats.append(channel.mean())
            feats.append(channel.std())
    return feats  # 12 values


def _color_histogram(img_hsv, bins=8):
    feats = []
    for c in range(3):
        hist = cv2.calcHist([img_hsv], [c], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.extend(hist.tolist())
    return feats  # 24 values


def _glcm_features(gray):
    # Reduce gray levels for a stable, fast GLCM
    gray_q = (gray / 32).astype(np.uint8)  # 0-7 levels
    glcm = graycomatrix(
        gray_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8, symmetric=True, normed=True
    )
    props = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]
    feats = [graycoprops(glcm, p).mean() for p in props]
    return feats  # 6 values


def _edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return [float(np.mean(edges > 0))]  # 1 value


def _hu_moments(gray):
    # Threshold to isolate darker lesion/spot regions from the leaf
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    # log-scale for numerical stability (Hu moments can be tiny)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.tolist()  # 7 values


def _lesion_ratio(img_hsv):
    # Rough heuristic: brown/yellow/dark spots typical of rice leaf disease
    lower = np.array([5, 40, 20])
    upper = np.array([35, 255, 200])
    mask = cv2.inRange(img_hsv, lower, upper)
    return [float(np.mean(mask > 0))]  # 1 value


def extract_features(image_bgr_or_rgb, is_bgr=True):
    """
    Take a single image (as loaded by cv2.imread -> BGR, or a PIL/np RGB
    array from Gradio) and return a 1D numpy feature vector.
    """
    img = image_bgr_or_rgb
    if img is None:
        raise ValueError("Empty image passed to extract_features")

    if not is_bgr:  # e.g. coming from PIL/Gradio as RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []
    feats += _color_stats(img_rgb, img_hsv)
    feats += _color_histogram(img_hsv)
    feats += _glcm_features(gray)
    feats += _edge_density(gray)
    feats += _hu_moments(gray)
    feats += _lesion_ratio(img_hsv)

    return np.array(feats, dtype=np.float32)


FEATURE_LENGTH = len(extract_features(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), is_bgr=True))
