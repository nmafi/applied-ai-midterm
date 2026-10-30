from pathlib import Path
import tensorflow as tf
from data_loader import load_dataset

tf.keras.utils.set_random_seed(42)

# Configure GPU memory
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

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

# Load best checkpoint from Stage 1
model = tf.keras.models.load_model(
    "checkpoints/model_a/best_head_continued.keras"
)

# Unfreeze last 30 MobileNetV2 layers
backbone = model.get_layer("mobilenetv2_1.00_128")
backbone.trainable = True

for layer in backbone.layers[:-30]:
    layer.trainable = False

for layer in backbone.layers[-30:]:
    layer.trainable = not isinstance(
        layer, tf.keras.layers.BatchNormalization
    )

# Recompile with smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

Path("checkpoints/model_a").mkdir(parents=True, exist_ok=True)
Path("results/model_a").mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_a/best_finetuned.keras",
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
        "results/model_a/finetune_history.csv"
    )
]

print("Trainable parameters:",
      sum(tf.size(w).numpy() for w in model.trainable_weights))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

model.save("checkpoints/model_a/final_finetuned.keras")

print("\nBest fine-tuning validation loss:",
      min(history.history["val_loss"]))

print("Best fine-tuning validation accuracy:",
      max(history.history["val_accuracy"]))
