"""
rice_gui_app.py
----------------
A simple, farmer-friendly GUI for the Rice Leaf Disease Random Forest model.

HOW TO RUN (in your Kaggle/Colab notebook, AFTER running train_rice_rf.py
so that rice_rf_model.joblib and label_encoder.joblib exist):

    !pip install gradio -q
    !python rice_gui_app.py

This launches a Gradio web app (with a public shareable link if share=True)
where a farmer can:
  1. Upload or take a photo of a rice leaf.
  2. Click "Diagnose".
  3. See the predicted disease (in English + Bangla), a confidence score,
     the top-3 most likely diseases, and simple, practical advice.

Make sure rice_features.py, rice_rf_model.joblib, and label_encoder.joblib
are all in the same working directory as this script (or edit MODEL_DIR below).
"""

import os
import numpy as np
import joblib
import gradio as gr

from rice_features import extract_features

MODEL_DIR = "/kaggle/working"   # <- change to wherever your .joblib files are
MODEL_PATH = os.path.join(MODEL_DIR, "rice_rf_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# ------------------------------------------------------------------
# Farmer-facing info per disease: Bangla name + short, practical advice.
# Edit freely to match your local agronomy recommendations.
# ------------------------------------------------------------------
DISEASE_INFO = {
    "Healthy Rice Leaf": {
        "bangla": "সুস্থ ধান পাতা",
        "advice": "Leaf looks healthy. No treatment needed — keep monitoring the field weekly.",
    },
    "Bacterial Leaf Blight": {
        "bangla": "ব্যাকটেরিয়াল লিফ ব্লাইট",
        "advice": "Drain excess standing water, avoid excess nitrogen fertilizer, and remove/burn "
                   "severely infected leaves. Consult your local agricultural office about approved "
                   "bactericides (e.g. copper-based sprays) for your region.",
    },
    "Brown Spot": {
        "bangla": "ব্রাউন স্পট (বাদামী দাগ রোগ)",
        "advice": "Often linked to potassium/nutrient deficiency or seed-borne infection. Balance "
                   "fertilizer (esp. potassium), use certified disease-free seed next season, and "
                   "consult local extension services about a suitable fungicide if severe.",
    },
    "Leaf Blast": {
        "bangla": "লিফ ব্লাস্ট",
        "advice": "Avoid excess nitrogen, ensure good field drainage, and consult local agricultural "
                   "office promptly about approved fungicides — blast can spread fast in humid weather.",
    },
    "Leaf scald": {
        "bangla": "লিফ স্ক্যাল্ড",
        "advice": "Common in dense, over-fertilized crops. Improve field spacing/airflow, avoid excess "
                   "nitrogen, and remove badly affected leaves.",
    },
    "Sheath Blight": {
        "bangla": "শীথ ব্লাইট",
        "advice": "Reduce plant density, avoid excess nitrogen, improve drainage, and consult local "
                   "extension office about approved fungicides if spreading in the field.",
    },
}

DEFAULT_ADVICE = "Consult your local agricultural extension office for a confirmed diagnosis and treatment."


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        raise FileNotFoundError(
            f"Could not find model files in {MODEL_DIR}.\n"
            f"Expected: {MODEL_PATH} and {ENCODER_PATH}.\n"
            f"Run train_rice_rf.py first, or update MODEL_DIR at the top of this file."
        )
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return clf, le


clf, le = load_artifacts()


def diagnose(image):
    if image is None:
        return "Please upload a photo of a rice leaf.", "", {}

    # Gradio gives us a numpy RGB array
    feats = extract_features(np.array(image), is_bgr=False).reshape(1, -1)

    proba = clf.predict_proba(feats)[0]
    order = np.argsort(proba)[::-1]
    top_idx = order[0]
    disease = le.classes_[top_idx]
    confidence = proba[top_idx]

    info = DISEASE_INFO.get(disease, {})
    bangla = info.get("bangla", "")
    advice = info.get("advice", DEFAULT_ADVICE)

    headline = f"🌾 {disease}" + (f"  ({bangla})" if bangla else "")
    headline += f"\nConfidence: {confidence * 100:.1f}%"

    top3 = {le.classes_[i]: float(proba[i]) for i in order[:3]}

    return headline, advice, top3


with gr.Blocks(title="Rice Leaf Disease Detector") as demo:
    gr.Markdown(
        "# 🌾 Rice Leaf Disease Detector\n"
        "Upload or take a photo of a rice leaf to check for common diseases. "
        "ধানের পাতার ছবি তুলুন বা আপলোড করুন — রোগ শনাক্ত করতে সাহায্য করবে।"
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Rice Leaf Photo / ধানের পাতার ছবি", sources=["upload", "webcam"])
            btn = gr.Button("🔍 Diagnose / নির্ণয় করুন", variant="primary")
        with gr.Column():
            result_label = gr.Textbox(label="Diagnosis / রোগ নির্ণয়", lines=3)
            advice_box = gr.Textbox(label="Advice / পরামর্শ", lines=4)
            top3_label = gr.Label(label="Top 3 possibilities / সম্ভাব্য রোগসমূহ")

    btn.click(fn=diagnose, inputs=image_input, outputs=[result_label, advice_box, top3_label])
    image_input.change(fn=diagnose, inputs=image_input, outputs=[result_label, advice_box, top3_label])

    gr.Markdown(
        "_Disclaimer: This is an AI-assisted tool and not a substitute for expert agronomic advice. "
        "For severe or spreading infections, please contact your local agricultural extension office._"
    )

if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives a public link, handy on Kaggle/Colab
