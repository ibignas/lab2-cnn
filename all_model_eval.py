import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

# 1. Setup Environment
mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(False)

IMG_SIZE = (224, 224)
BATCH_SIZE = 24
AUTOTUNE = tf.data.AUTOTUNE

# Models we want to compare (Must match the saved prefixes)
RUNS = ["vgg16_aug", "ResNet50", "MobileNetV2", "EfficientNetB0"]

# Preprocessing functions mapping
PREPROCESS_FNS = {
    "vgg16_aug": keras.applications.vgg16.preprocess_input,
    "ResNet50": keras.applications.resnet50.preprocess_input,
    "MobileNetV2": keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": keras.applications.efficientnet.preprocess_input
}

# 2. Base Test Dataset (No preprocessing yet)
test_ds_base = keras.utils.image_dataset_from_directory(
    "data/test", shuffle=False, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

comparison_results = []

print("========== Generating Comparison Table from Saved Models ==========\n")

for run in RUNS:
    model_path = Path(f"models/{run}_best.keras")
    meta_path = Path(f"results/metadata/{run}_metadata.json")
    hist_path = Path(f"results/history/{run}_history.json")

    if not (model_path.exists() and meta_path.exists() and hist_path.exists()):
        print(f"Skipping {run}: Missing model, metadata, or history file.")
        continue

    print(f"Evaluating {run}...")

    # Load Model
    model = keras.models.load_model(model_path)

    # Load Metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
        train_time = meta.get("train_time_seconds", 0)

    # Load History to get Best Validation metrics
    with open(hist_path, "r") as f:
        data = json.load(f)
        hist = data.get("history", data)  # Safely handle dict format

    val_loss = min(hist.get('val_loss', [0]))
    val_acc = max(hist.get('val_accuracy', [0]))

    # Prepare Test Dataset with specific preprocessing
    preprocess_fn = PREPROCESS_FNS[run]
    test_ds = test_ds_base.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # Measure Inference Time
    inf_start = time.perf_counter()
    model.predict(test_ds, verbose=0)
    inf_time = time.perf_counter() - inf_start

    # Measure Test Loss and Accuracy
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Calculate Trainable Params (Temporarily unfreeze backbone locally to count correctly)
    total_params = model.count_params()

    # Find the backbone layer (usually the second layer, index 1)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # The Functional API backbone
            backbone = layer
            break

    if backbone:
        backbone.trainable = True
        # Emulate the unfreezing logic used during fine-tuning
        cutoff = -4 if "vgg" in run else -20
        for l in backbone.layers[:cutoff]:
            l.trainable = False

    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])

    comparison_results.append({
        "Model": run.replace("vgg16_aug", "VGG16 (Aug)"),
        "Total Params": f"{total_params:,}",
        "Trainable Params": f"{trainable_params:,}",
        "Train Time (s)": round(train_time, 2),
        "Inference Time (s)": round(inf_time, 3),
        "Val Acc": round(val_acc, 4),
        "Test Acc": round(test_acc, 4),
        "Val Loss": round(val_loss, 4),
        "Test Loss": round(test_loss, 4)
    })

# 3. Print and Save the Table
print("\n\n========== TASK 5: FINAL COMPARISON TABLE ==========\n")
df = pd.DataFrame(comparison_results)
markdown_table = df.to_markdown(index=False)

print(markdown_table)

# Save to a Markdown file
EVAL_DIR = Path("evaluation_results")
EVAL_DIR.mkdir(exist_ok=True)
with open(EVAL_DIR / "model_comparison.md", "w") as f:
    f.write("# Model Comparison Results\n\n")
    f.write(markdown_table)

print("\nSaved comparison table to evaluation_results/model_comparison.md")