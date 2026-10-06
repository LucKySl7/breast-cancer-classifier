# Breast Cancer Histopathology Classifier

A convolutional neural network (CNN) built with TensorFlow/Keras to classify breast tissue histopathology image patches as cancerous (IDC-positive) or benign, deployed as an interactive Streamlit app for live predictions.

## Overview

- Trained a CNN (CancerNet architecture) on ~277K histology image patches
- Achieved 73% classification accuracy on the test set
- Built an interactive Streamlit web app that loads the trained model and returns live predictions on user-uploaded images

## Project structure

```
.
├── app.py                   # Streamlit app for live predictions
├── build_dataset.py         # Splits the raw dataset into training/validation/testing sets
├── train_model.py           # Trains the CNN and saves the model + training plot
├── cancernet_model.keras    # Trained model weights
├── plot.png                 # Training accuracy/loss curves
├── pyimagesearch/
│   ├── cancernet.py         # CNN architecture definition
│   └── config.py            # Paths and training configuration
├── requirements.txt
└── datasets/                # Not included — see "Dataset" below
```


## Dataset

The dataset (~277K image patches, organized into training/validation/testing splits) is not included in this repo due to its size. This project uses the [Breast Histopathology Images dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) from Kaggle, which contains positive (IDC-positive) and negative (benign) patches sorted by patient. If you want to retrain the model, download the dataset from Kaggle, place it under `datasets/orig`, and run `build_dataset.py` to generate the training/validation/testing splits.

## Setup

```bash
git clone https://github.com/LucKySl7/breast-cancer-classifier.git
cd breast-cancer-classifier

python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running the app

The trained model (`cancernet_model.keras`) is included in the repo, so you can run the app directly without retraining:

```bash
streamlit run app.py
```

Upload a histology image patch and the app will return a live prediction.

## Retraining the model

```bash
python build_dataset.py
python train_model.py
```

This regenerates `cancernet_model.keras` and `plot.png`.

## What I'd add next

- Improve accuracy beyond 73% through data augmentation and hyperparameter tuning
- Add confidence scores alongside predictions in the app
- Expand the app to support batch image uploads

## What this project demonstrates

Built while learning deep learning fundamentals — convolutional neural networks, training/validation/test splits, and deploying a trained model behind a usable interface with Streamlit.