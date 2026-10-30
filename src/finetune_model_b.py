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
    "checkpoints/model_b/best_head_continued.keras"
)

backbone = model.get_layer("mobilenetv2_1.00_128")
backbone.trainable = True

for layer in backbone.layers[:-30]:
    layer.trainable = False

for layer in backbone.layers[-30:]:
    layer.trainable = not isinstance(
        layer, tf.keras.layers.BatchNormalization
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/model_b/best_finetuned.keras",
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
        "results/model_b/finetune_history.csv"
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

model.save("checkpoints/model_b/final_finetuned.keras")
