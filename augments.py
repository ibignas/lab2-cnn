from tensorflow import keras
import tensorflow as tf
from pathlib import Path

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", tf.config.list_physical_devices("GPU"))

tf.config.optimizer.set_jit(False)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
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

NUM_CLASSES = 6
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
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

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

    return [best_ckpt, last_ckpt, early_stop]

def train_model(run, train_data, val_data, initial_epochs=10, fine_tune_epochs=10):
    model, base_model = build_model()

    callbacks = get_callbacks(run)

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

    return model, history_head, history_finetune

Path("models").mkdir(exist_ok=True)

train_ds_plain = prepare(train_ds, training=True, augment=False)
val_ds_final = prepare(val_ds, training=False, augment=False)
test_ds_final = prepare(test_ds, training=False, augment=False)

vgg_plain_model, plain_head_hist, plain_ft_hist = train_model(
    run="vgg16_plain",
    train_data=train_ds_plain,
    val_data=val_ds_final,
    initial_epochs=10,
    fine_tune_epochs=10
)

train_ds_aug = prepare(train_ds, training=True, augment=True)

vgg_aug_model, aug_head_hist, aug_ft_hist = train_model(
    run="vgg16_aug",
    train_data=train_ds_aug,
    val_data=val_ds_final,
    initial_epochs=10,
    fine_tune_epochs=10
)