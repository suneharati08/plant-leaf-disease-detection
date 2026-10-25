# 🌿 Plant Leaf Disease Recognition System

An AI-powered web application that detects plant leaf diseases from images using deep learning (CNN) built with TensorFlow/Keras and Streamlit.

---

## 📌 Project Overview

This project provides an automated tool to assist farmers, agricultural researchers, and plant enthusiasts in identifying crop diseases early. By uploading a photo of a plant leaf, the system classifies the disease and provides detailed actionable recommendations, including symptoms, severity, and treatments.

### Key Features:
- **CNN Deep Learning Model**: Trained on thousands of plant leaf images to classify 38 different plant disease categories and healthy leaves.
- **Interactive Web Interface**: Clean, responsive Streamlit user interface with dark mode styling.
- **Disease Insights & Guidance**: Displays confidence scores, affected parts, severity levels, prevention tips, and chemical/organic treatments.
- **Jupyter Notebook Included**: Contains full model architecture, data augmentation, training pipeline, and evaluation metrics (`appleplant.ipynb`).

---

## 📊 Dataset

The model is trained on the **New Plant Diseases Dataset** available on Kaggle:
- [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

It contains ~87,000 RGB images of healthy and diseased crop leaves categorized into 38 distinct classes.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/suneharati08/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Setup (`trained_plant_disease_model.keras`)
> **Note**: Due to GitHub's 100 MB file size limit, the trained model file (`trained_plant_disease_model.keras`, ~274 MB) is excluded from the repository.

To get the model file:
1. Open and run the included notebook [`appleplant.ipynb`](appleplant.ipynb) to train the model on the dataset.
2. The notebook will automatically save `trained_plant_disease_model.keras` in the root project directory.
3. Once generated, place `trained_plant_disease_model.keras` directly in the project root directory.

### 5. Run the Streamlit Application
```bash
streamlit run main.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
├── appleplant.ipynb      # Model training & evaluation Jupyter notebook
├── main.py               # Streamlit web application
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignore configuration (ignores .keras model binaries)
├── th.jpg                # Sample preview image
└── README.md             # Project documentation
```

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** - CNN model architecture and inference
- **Streamlit** - Interactive web UI
- **NumPy & Pillow** - Image preprocessing and transformation
