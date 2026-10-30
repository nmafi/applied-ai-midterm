from pathlib import Path
import tensorflow as tf

from srgan_models import build_generator
from srgan_data import load_srgan_dataset

tf.keras.utils.set_random_seed(42)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

Path("checkpoints/srgan").mkdir(parents=True, exist_ok=True)
Path("results/srgan").mkdir(parents=True, exist_ok=True)

train_ds = load_srgan_dataset(
    "data/splits/train_final.csv",
    batch_size=16,
    training=True
)

val_ds = load_srgan_dataset(
    "data/splits/val.csv",
    batch_size=16,
    training=False
)

generator = build_generator()

generator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae")
    ]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/srgan/best_pretrained.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        "results/srgan/pretrain_history.csv"
    )
]

print("\nStarting generator pretraining...\n")

history = generator.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

generator.save("checkpoints/srgan/pretrained_generator.keras")

print("\nBest validation MSE:", min(history.history["val_loss"]))
