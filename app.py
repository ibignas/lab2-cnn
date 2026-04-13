import os
import io
import re
import json
import random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --- STRICT LOCAL CPU INFERENCE REQUIREMENT ---
tf.config.set_visible_devices([], 'GPU')

IMG_SIZE = (224, 224)

# Set Streamlit Page Config
st.set_page_config(page_title="Vehicle Classifier", layout="wide")


# --- READ THE COMPARISON TABLE TO FIND THE BEST MODEL ---
@st.cache_data
def get_model_rankings():
    table_path = "evaluation_results/model_comparison.md"
    default_models = ["EfficientNetB0", "MobileNetV2", "ResNet50", "vgg16_aug"]

    if not os.path.exists(table_path):
        return default_models

    with open(table_path, "r") as f:
        markdown_content = f.read()

    # Extract just the table part of the markdown file
    table_match = re.search(r'(\|.*\|[\r\n]+)+', markdown_content)
    if not table_match:
        return default_models

    table_str = table_match.group(0)

    try:
        # Use Pandas to read the markdown table
        # The separator is '|', we drop the empty first and last columns created by the pipes, and skip the markdown separator row
        df = pd.read_table(io.StringIO(table_str), sep="|", header=0, skipinitialspace=True).dropna(axis=1,
                                                                                                    how='all').iloc[1:]

        # Strip whitespace from column names and string values
        df.columns = df.columns.str.strip()
        df['Model'] = df['Model'].str.strip()

        # Convert accuracy columns back to floats so we can sort them
        df['Test Acc'] = df['Test Acc'].astype(float)
        df['Val Acc'] = df['Val Acc'].astype(float)

        # Sort primarily by Test Accuracy, then Validation Accuracy
        df = df.sort_values(by=['Test Acc', 'Val Acc'], ascending=[False, False])

        # Extract the sorted model names and format VGG16 back to its filename
        ranked_models = df['Model'].tolist()
        ranked_models = ["vgg16_aug" if m == "VGG16 (Aug)" else m for m in ranked_models]

        return ranked_models
    except Exception:
        return default_models


# Get the dynamically sorted list
AVAILABLE_MODELS = get_model_rankings()
DYNAMIC_BEST_MODEL = AVAILABLE_MODELS[0]

# --- SIDEBAR MODEL SELECTION ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"🏆 **Auto-detected Best Model:** `{DYNAMIC_BEST_MODEL}`")

# The selectbox defaults to index 0, which is now guaranteed to be the highest accuracy model
SELECTED_MODEL = st.sidebar.selectbox(
    "Choose a Model for Inference:",
    options=AVAILABLE_MODELS,
    index=0
)


# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_model(model_name):
    model_path = f"models/{model_name}_best.keras"
    return keras.models.load_model(model_path)


@st.cache_data
def load_class_names(model_name):
    # Attempt to read from metadata first
    meta_path = f"results/metadata/{model_name}_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "class_names" in meta:
                return meta["class_names"]

    # Fallback: Read directly from the train directory folders
    train_dir = "data/train"
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return classes


def get_preprocess_function(model_name):
    if "EfficientNet" in model_name:
        return keras.applications.efficientnet.preprocess_input
    elif "MobileNet" in model_name:
        return keras.applications.mobilenet_v2.preprocess_input
    elif "ResNet" in model_name:
        return keras.applications.resnet50.preprocess_input
    else:
        return keras.applications.vgg16.preprocess_input


# Load resources based on the dropdown selection
try:
    model = load_model(SELECTED_MODEL)
    class_names = load_class_names(SELECTED_MODEL)
    preprocess_fn = get_preprocess_function(SELECTED_MODEL)
except Exception as e:
    st.error(f"Error loading model or metadata. Ensure you have trained {SELECTED_MODEL}. Details: {e}")
    st.stop()

# --- AUGMENTATION PIPELINE (For Visualization) ---
data_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomTranslation(0.08, 0.08),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.15),
])

# --- UI LAYOUT ---
st.title("🚗 Vehicle Classification AI")
st.markdown(f"**Currently Active Model:** `{SELECTED_MODEL}` (Running via Local CPU Inference)")

# Create Tabs
tab_predict, tab_visuals, tab_metrics = st.tabs([
    "🎯 Run Inference",
    "🖼️ Dataset & Augmentations",
    "📊 Model Metrics & Comparison"
])

# ================= TAB 1: INFERENCE =================
with tab_predict:
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # Display Uploaded Image
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

        # Run Inference
        with col2:
            st.subheader("Prediction Results")
            with st.spinner(f'Running Local CPU Inference with {SELECTED_MODEL}...'):
                # Preprocess
                img_resized = image.resize(IMG_SIZE)
                img_array = keras.utils.img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_fn(img_batch)

                # Predict
                predictions = model.predict(img_preprocessed)
                confidence = np.max(predictions)
                predicted_class = class_names[np.argmax(predictions)]

            # Display Results
            st.success(f"**Predicted Class:** {predicted_class.upper()}")
            st.info(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(float(confidence))

            # Show all probabilities
            st.write("---")
            st.write("**Probabilities for all classes:**")
            prob_df = pd.DataFrame({
                "Class": class_names,
                "Probability": predictions[0]
            }).sort_values(by="Probability", ascending=False)
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

# ================= TAB 2: VISUALIZATIONS =================
with tab_visuals:
    st.header("Dataset & Augmentation Visualization")
    st.write("Visualizing how the dataset looks before and after the training augmentation pipeline is applied.")

    try:
        test_dir = "data/test"
        random_class = random.choice(class_names)
        class_dir = os.path.join(test_dir, random_class)
        random_img_name = random.choice(os.listdir(class_dir))
        sample_img_path = os.path.join(class_dir, random_img_name)

        sample_img = Image.open(sample_img_path).convert("RGB").resize(IMG_SIZE)
        sample_array = keras.utils.img_to_array(sample_img)
        sample_batch = np.expand_dims(sample_array, axis=0)

        augmented_batch = data_augment(sample_batch, training=True)
        augmented_img = keras.utils.array_to_img(augmented_batch[0])

        col_orig, col_aug = st.columns(2)
        with col_orig:
            st.subheader(f"Original Dataset Image ({random_class})")
            st.image(sample_img, use_container_width=True)
        with col_aug:
            st.subheader("After Augmentation Pipeline")
            st.image(augmented_img, use_container_width=True)
            st.caption("Applied: Random Horizontal Flip, Rotation, Translation, Zoom, Contrast.")

    except Exception as e:
        st.warning(
            "Could not load sample images from `data/test`. Ensure the data folder is in the same directory as this script.")

# ================= TAB 3: METRICS =================
with tab_metrics:
    st.header("Model Performance & Comparison")

    # 1. Show Comparison Table
    st.subheader("1. All Models Prediction Results & Comparison")
    table_path = "evaluation_results/model_comparison.md"
    if os.path.exists(table_path):
        with open(table_path, "r") as f:
            st.markdown(f.read())
    else:
        st.warning("Comparison table not found. Run the Task 5 script first.")

    # 2. Show Metrics for the Selected Model
    st.write("---")
    st.subheader(f"2. {SELECTED_MODEL} Metrics")

    col_cm, col_lc = st.columns(2)
    with col_cm:
        st.write("**Confusion Matrix**")
        cm_path = f"evaluation_results/{SELECTED_MODEL}_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.write("Confusion matrix not found.")

    with col_lc:
        st.write("**Learning Curves**")
        lc_path = f"evaluation_results/{SELECTED_MODEL}_learning_curves.png"
        if os.path.exists(lc_path):
            st.image(lc_path, use_container_width=True)
        else:
            st.write("Learning curves not found.")

    st.write("**Classification Report (Precision, Recall, F1-Score)**")
    report_path = f"evaluation_results/{SELECTED_MODEL}_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            st.code(f.read())