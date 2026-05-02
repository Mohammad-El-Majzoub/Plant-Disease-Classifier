import gradio as gr
import tensorflow as tf
import numpy as np
import json

# ── Load model and class names ──────────────────────────────────────────────
model = tf.keras.models.load_model("EffNetB0.keras")

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]

IMG_SIZE = (224, 224)

# ── Prediction function ─────────────────────────────────────────────────────
def classify_leaf(image):
    if image is None:
        return {"No image uploaded": 1.0}

    img = tf.image.resize(image, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    probs = model.predict(img, verbose=0)[0]

    top5_idx = np.argsort(probs)[::-1][:5]
    results = {}
    for idx in top5_idx:
        label = class_names[idx].replace("___", " - ").replace("_", " ")
        results[label] = float(probs[idx])

    return results

# ── Gradio Interface ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=classify_leaf,
    inputs=gr.Image(type="numpy", label="Upload a Leaf Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="Plant Disease Classifier",
    description=(
        "Upload a photo of a plant leaf to identify the disease. "
        "This model classifies 38 types of plant diseases and healthy leaves "
        "using EfficientNetB0, trained on 54,305 images from the PlantVillage dataset "
        "with 99.41% test accuracy."
    ),
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
