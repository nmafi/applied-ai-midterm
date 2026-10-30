import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset(
    csv_path,
    image_size=(128, 128),
    batch_size=32,
    training=False
):
    df = pd.read_csv(csv_path)

    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype("float32").tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    def preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = preprocess_input(tf.cast(image, tf.float32))
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds
