from pathlib import Path
import tensorflow as tf

from data_loader_b_png import load_dataset
from model_a import build_model_a

tf.keras.utils.set_random_seed(42)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

train_ds = load_dataset(
    "data/splits/train_final_srgan.csv",
    batch_size=32,
    training=True
)

val_ds = load_dataset(
    "data/splits/val_srgan.csv",
    batch_size=32,
    training=False
)

# Fresh ImageNet-pretrained model, not Model A's trained weights.
model = build_model_a()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_b/best_head.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.CSVLogger(
        "results/model_b/head_history.csv"
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model.save("checkpoints/model_b/final_head.keras")
