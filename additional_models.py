import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Setup Environment
mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(False)

IMG_SIZE = (224, 224)
BATCH_SIZE = 24
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

Path("models").mkdir(exist_ok=True)
Path("results/history").mkdir(parents=True, exist_ok=True)
Path("results/metadata").mkdir(parents=True, exist_ok=True)
EVAL_DIR = Path("evaluation_results")
EVAL_DIR.mkdir(exist_ok=True)

# 2. Base Datasets (No preprocessing yet, so we can apply model-specific ones)
train_ds_base = keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)
val_ds_base = keras.utils.image_dataset_from_directory(
    "data/train", validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=False
)
test_ds_base = keras.utils.image_dataset_from_directory(
    "data/test", shuffle=False, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

class_names = train_ds_base.class_names
NUM_CLASSES = len(class_names)
INPUT_SHAPE = (224, 224, 3)

data_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomTranslation(0.08, 0.08),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.15),
], name="data_augment")


def get_callbacks(run):
    return [
        keras.callbacks.ModelCheckpoint(f"models/{run}_best.keras", monitor="val_loss", save_best_only=True, mode="min",
                                        verbose=1),
        keras.callbacks.ModelCheckpoint(f"models/{run}_last.keras", save_weights_only=False, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, mode="min",
                                          min_lr=1e-7)
    ]


# Models configuration
models_to_train = {
    "ResNet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "MobileNetV2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input)
}

comparison_results = []

# ================= TASK 4: Adapt and Train New Models =================
for name, (ModelClass, preprocess_fn) in models_to_train.items():
    print(f"\n{'=' * 20} Training {name} {'=' * 20}")

    # Apply model-specific preprocessing and augmentation (Augment Training ONLY)
    train_ds = train_ds_base.map(lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(
        AUTOTUNE)

    val_ds = val_ds_base.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds_base.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # Build Model
    base_model = ModelClass(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)

    # Calculate params
    total_params = model.count_params()
    base_model.trainable = True
    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    base_model.trainable = False  # Re-freeze for phase 1

    # Phase 1: Train Head
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start_time = time.perf_counter()
    hist_head = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=get_callbacks(name))

    # Phase 2: Fine-Tuning (Unfreeze last 20 layers - generic enough for these models)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist_ft = model.fit(train_ds, validation_data=val_ds, epochs=20, initial_epoch=hist_head.epoch[-1] + 1,
                        callbacks=get_callbacks(name))
    train_time = time.perf_counter() - start_time

    # Save artifacts
    merged_hist = {k: hist_head.history.get(k, []) + hist_ft.history.get(k, []) for k in
                   set(hist_head.history) | set(hist_ft.history)}
    pd.DataFrame(merged_hist).to_csv(f"results/history/{name}_history.csv", index_label="epoch")
    with open(f"results/history/{name}_history.json", "w") as f:
        json.dump({"history": merged_hist, "fine_tune_start_epoch": len(hist_head.history["loss"])}, f)
    with open(f"results/metadata/{name}_metadata.json", "w") as f:
        json.dump({"train_time_seconds": train_time}, f)

    # --- Task 3 Evaluation for new models ---
    best_model = keras.models.load_model(f"models/{name}_best.keras")

    # Plot curves
    epochs = range(1, len(merged_hist['accuracy']) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, merged_hist['accuracy'], 'b', label='Train Acc')
    plt.plot(epochs, merged_hist['val_accuracy'], 'r', label='Val Acc')
    plt.axvline(x=len(hist_head.history['loss']), color='k', linestyle='--')
    plt.title(f'{name} - Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, merged_hist['loss'], 'b', label='Train Loss')
    plt.plot(epochs, merged_hist['val_loss'], 'r', label='Val Loss')
    plt.axvline(x=len(hist_head.history['loss']), color='k', linestyle='--')
    plt.title(f'{name} - Loss')
    plt.legend()
    plt.savefig(EVAL_DIR / f"{name}_learning_curves.png", dpi=200)
    plt.close()

    # Inference & Metrics
    inf_start = time.perf_counter()
    y_prob = best_model.predict(test_ds, verbose=0)
    inf_time = time.perf_counter() - inf_start

    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    val_loss = min(merged_hist['val_loss'])
    val_acc = max(merged_hist['val_accuracy'])

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    with open(EVAL_DIR / f"{name}_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues",
                                            xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{name}_confusion_matrix.png", dpi=200)
    plt.close()

    comparison_results.append({
        "Model": name,
        "Total Params": f"{total_params:,}",
        "Trainable Params": f"{trainable_params:,}",
        "Train Time (s)": round(train_time, 2),
        "Inference Time (s)": round(inf_time, 3),
        "Val Acc": round(val_acc, 4),
        "Test Acc": round(test_acc, 4),
        "Val Loss": round(val_loss, 4),
        "Test Loss": round(test_loss, 4)
    })

# ================= TASK 5: Add VGG16_Aug and Output Table =================
print("\nEvaluating previously trained VGG16_Aug to include in comparison...")
vgg_prep = keras.applications.vgg16.preprocess_input
vgg_val_ds = val_ds_base.map(lambda x, y: (vgg_prep(x), y), num_parallel_calls=AUTOTUNE)
vgg_test_ds = test_ds_base.map(lambda x, y: (vgg_prep(x), y), num_parallel_calls=AUTOTUNE)

vgg_model = keras.models.load_model("models/vgg16_aug_best.keras")
with open("results/metadata/vgg16_aug_metadata.json", "r") as f:
    vgg_meta = json.load(f)
with open("results/history/vgg16_aug_history.json", "r") as f:
    data = json.load(f)
    vgg_hist = data["history"] if "history" in data else data

inf_start = time.perf_counter()
vgg_model.predict(vgg_test_ds, verbose=0)
vgg_inf = time.perf_counter() - inf_start

vgg_test_loss, vgg_test_acc = vgg_model.evaluate(vgg_test_ds, verbose=0)

# Unfreeze VGG locally just to count trainable params as they were during fine-tuning
vgg_base = vgg_model.layers[1]
vgg_base.trainable = True
for l in vgg_base.layers[:-4]: l.trainable = False

comparison_results.insert(0, {
    "Model": "VGG16 (Aug)",
    "Total Params": f"{vgg_model.count_params():,}",
    "Trainable Params": f"{sum([keras.backend.count_params(w) for w in vgg_model.trainable_weights]):,}",
    "Train Time (s)": round(vgg_meta.get("train_time_seconds", 0), 2),
    "Inference Time (s)": round(vgg_inf, 3),
    "Val Acc": round(max(vgg_hist['val_accuracy']), 4),
    "Test Acc": round(vgg_test_acc, 4),
    "Val Loss": round(min(vgg_hist['val_loss']), 4),
    "Test Loss": round(vgg_test_loss, 4)
})

print("\n\n========== TASK 5: FINAL COMPARISON TABLE ==========\n")
df = pd.DataFrame(comparison_results)
print(df.to_markdown(index=False))