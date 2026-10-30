import tensorflow as tf
from data_loader_b_png import load_dataset

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

model = tf.keras.models.load_model(
    "checkpoints/model_b/best_head.keras"
)

print(
    "Trainable parameters:",
    sum(tf.size(w).numpy() for w in model.trainable_weights)
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_b/best_head_continued.keras",
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
        "results/model_b/head_continued_history.csv"
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=10,
    epochs=20,
    callbacks=callbacks
)

model.save(
    "checkpoints/model_b/final_head_continued.keras"
)
