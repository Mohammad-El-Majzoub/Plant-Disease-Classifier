---
title: Plant Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
---

# Plant Disease Classifier

A deep learning model that classifies 38 types of plant diseases and healthy leaves from leaf images.

## Model Details
- **Architecture:** EfficientNetB0 (fine-tuned)
- **Dataset:** PlantVillage (54,305 images, 38 classes)
- **Test Accuracy:** 99.41%
- **F1 Score:** 0.9922
- **AUC:** 0.9999

## How to Use
Upload a photo of a plant leaf and the model will predict the disease (or healthy status) with confidence scores for the top 5 predictions.

## Tech Stack
Python, TensorFlow, Keras, Gradio

## Live Demo

A live version of this Plant Disease Classifier is deployed on Hugging Face Spaces:

https://mohammad-el-majzoub-plant-disease-classifier.hf.space