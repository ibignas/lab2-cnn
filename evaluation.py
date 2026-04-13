import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Match Training Environment Settings
mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.optimizer.set_jit(False)

IMG_SIZE = (224, 224)
BATCH_SIZE = 24  # Matches your latest training script
AUTOTUNE = tf.data.AUTOTUNE
EVAL_DIR = Path("evaluation_results")
EVAL_DIR.mkdir(exist_ok=True)

RUNS = ["vgg16_plain", "vgg16_aug"]


# 2. Test Dataset Preparation
def prepare_test_dataset():
    test_ds = keras.utils.image_dataset_from_directory(
        "data/test",
        shuffle=False,  # Essential: Keep False to map predictions correctly to true labels
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb"
    )
    class_names = test_ds.class_names

    preprocess = keras.applications.vgg16.preprocess_input
    test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return test_ds, class_names


# 3. Plotting Training and Validation Curves
def plot_learning_curves(run):
    history_path = Path(f"results/history/{run}_history.json")
    if not history_path.exists():
        print(f"History file not found for {run}. Skipping plots.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your script saves the raw history dict directly, not wrapped in a "history" key.
    # Adjusting based on how merge_histories output is dumped.
    if "history" in data:
        history = data["history"]
        ft_start = data.get("fine_tune_start_epoch", 10)
    else:
        # If it was saved directly as the dict
        history = data
        # Fallback to 10 if metadata not found inside the dict
        ft_start = 10

        # Read metadata separately to guarantee we have ft_start
    metadata_path = Path(f"results/metadata/{run}_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            ft_start = meta.get("fine_tune_start_epoch", ft_start)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.axvline(x=ft_start, color='k', linestyle='--', label='Start Fine-Tuning')
    plt.title(f'{run} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.axvline(x=ft_start, color='k', linestyle='--', label='Start Fine-Tuning')
    plt.title(f'{run} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_out = EVAL_DIR / f"{run}_learning_curves.png"
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved learning curves to {plot_out}")


# 4. Evaluating Precision, Recall, F1-Score, and Confusion Matrix
def evaluate_model(run, test_ds, class_names):
    model_path = Path(f"models/{run}_best.keras")
    if not model_path.exists():
        print(f"Model file not found for {run}. Skipping evaluation.")
        return

    print(f"\n--- Evaluating {run} ---")
    model = keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(report)

    report_out = EVAL_DIR / f"{run}_classification_report.txt"
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(f"Model: {run}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix - {run}")
    plt.tight_layout()

    cm_out = EVAL_DIR / f"{run}_confusion_matrix.png"
    plt.savefig(cm_out, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {cm_out}")


# 5. Main Execution Block
def main():
    test_ds, class_names = prepare_test_dataset()
    print("Testing on classes:", class_names)

    for run in RUNS:
        plot_learning_curves(run)
        evaluate_model(run, test_ds, class_names)


if __name__ == "__main__":
    main()