from pathlib import Path
import tensorflow as tf
from data_loader import load_dataset

tf.keras.utils.set_random_seed(42)

train_ds = load_dataset(
    "data/splits/train_final.csv",
    batch_size=32,
    training=True
)

val_ds = load_dataset(
    "data/splits/val.csv",
    batch_size=32,
    training=False
)

model = tf.keras.models.load_model(
    "checkpoints/model_a/best_head.keras"
)

print("Trainable parameters:",
      sum(tf.size(w).numpy() for w in model.trainable_weights))

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_a/best_head_continued.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.CSVLogger(
        "results/model_a/head_continued_history.csv"
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=10,
    epochs=20,
    callbacks=callbacks
)

model.save("checkpoints/model_a/final_head_continued.keras")
