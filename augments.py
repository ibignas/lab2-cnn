from tensorflow import keras
from tensorflow.keras import mixed_precision
import tensorflow as tf
from pathlib import Path
import json
import time
import pandas as pd

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", tf.config.list_physical_devices("GPU"))

tf.config.optimizer.set_jit(False)

IMG_SIZE = (224, 224)
BATCH_SIZE = 24
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    "data/test",
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)


data_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomTranslation(0.08, 0.08),
    keras.layers.RandomZoom(0.10),
    keras.layers.RandomContrast(0.15),
], name="data_augment")

preprocess = keras.applications.vgg16.preprocess_input

def prepare(ds, training=False, augment=False):
    if training and augment:
        ds = ds.map(
            lambda x, y: (data_augment(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=AUTOTUNE
    )

    if training:
        ds = ds.shuffle(1000)

    return ds.prefetch(AUTOTUNE)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
INPUT_SHAPE = (224, 224, 3)

def build_model(num_classes=NUM_CLASSES):
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

def get_callbacks(run):
    best_ckpt = keras.callbacks.ModelCheckpoint(
        filepath=f"models/{run}_best.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    last_ckpt = keras.callbacks.ModelCheckpoint(
        filepath=f"models/{run}_last.keras",
        save_weights_only=False,
        verbose=1
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1,
        mode="min",
        min_lr=1e-7
    )

    return [best_ckpt, last_ckpt, early_stop, reduce_lr]

def merge_histories(history_head, history_finetune):
    merged = {}
    keys = set(history_head.history.keys()) | set(history_finetune.history.keys())

    for key in keys:
        head_values = history_head.history.get(key, [])
        ft_values = history_finetune.history.get(key, [])
        merged[key] = head_values + ft_values

    return merged


def save_run_artifacts(run, history_head, history_finetune, train_time_seconds, class_names):
    merged_history = merge_histories(history_head, history_finetune)
    fine_tune_start_epoch = len(history_head.history.get("loss", []))

    history_json_path = Path("results/history") / f"{run}_history.json"
    history_csv_path = Path("results/history") / f"{run}_history.csv"
    metadata_json_path = Path("results/metadata") / f"{run}_metadata.json"

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    history_df = pd.DataFrame(merged_history)
    history_df.index.name = "epoch"
    history_df.to_csv(history_csv_path)

    metadata = {
        "run_name": run,
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "img_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "fine_tune_start_epoch": fine_tune_start_epoch,
        "total_epochs": len(merged_history.get("loss", [])),
        "train_time_seconds": train_time_seconds,
        "best_model_path": f"models/{run}_best.keras",
        "last_model_path": f"models/{run}_last.keras",
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def train_model(run, train_data, val_data, class_names, initial_epochs=10, fine_tune_epochs=10):
    model, base_model = build_model()
    callbacks = get_callbacks(run)

    start_time = time.perf_counter()

    history_head = model.fit(
        train_data,
        validation_data=val_data,
        epochs=initial_epochs,
        callbacks=callbacks
    )

    base_model.trainable = True

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_finetune = model.fit(
        train_data,
        validation_data=val_data,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=history_head.epoch[-1] + 1,
        callbacks=callbacks
    )

    train_time_seconds = time.perf_counter() - start_time

    save_run_artifacts(
        run=run,
        history_head=history_head,
        history_finetune=history_finetune,
        train_time_seconds=train_time_seconds,
        class_names=class_names
    )

    return model, history_head, history_finetune

Path("models").mkdir(exist_ok=True)

Path("results").mkdir(exist_ok=True)
Path("results/history").mkdir(parents=True, exist_ok=True)
Path("results/metadata").mkdir(parents=True, exist_ok=True)

train_ds_plain = prepare(train_ds, training=True, augment=False)
val_ds_final = prepare(val_ds, training=False, augment=False)
test_ds_final = prepare(test_ds, training=False, augment=False)

vgg_plain_model, plain_head_hist, plain_ft_hist = train_model(
    run="vgg16_plain",
    train_data=train_ds_plain,
    val_data=val_ds_final,
    class_names=class_names,
    initial_epochs=10,
    fine_tune_epochs=20
)

train_ds_aug = prepare(train_ds, training=True, augment=True)

vgg_aug_model, aug_head_hist, aug_ft_hist = train_model(
    run="vgg16_aug",
    train_data=train_ds_aug,
    val_data=val_ds_final,
    class_names=class_names,
    initial_epochs=10,
    fine_tune_epochs=20
)