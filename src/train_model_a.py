import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from data_loader import load_dataset
from model_a import build_model_a

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Avoid allocating all GPU memory immediately.
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

Path("checkpoints/model_a").mkdir(parents=True, exist_ok=True)
Path("results/model_a").mkdir(parents=True, exist_ok=True)

train_ds = load_dataset(
    "data/splits/train_final.csv",
    image_size=(128, 128),
    batch_size=32,
    training=True
)

val_ds = load_dataset(
    "data/splits/val.csv",
    image_size=(128, 128),
    batch_size=32,
    training=False
)

model = build_model_a()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_a/best_head.keras",
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
        "results/model_a/head_history.csv"
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model.save("checkpoints/model_a/final_head.keras")

print("\nBest validation accuracy:",
      max(history.history["val_accuracy"]))

print("Best validation AUC:",
      max(history.history["val_auc"]))
