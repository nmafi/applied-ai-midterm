import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset_b(
    csv_path,
    generator,
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
        image = tf.image.decode_jpeg(image, channels=3)

        # Match SRGAN training preprocessing.
        hr = tf.image.resize(
            image,
            (128, 128),
            method="bicubic",
            antialias=True
        )
        hr = tf.cast(hr, tf.float32) / 127.5 - 1.0

        lr = tf.image.resize(
            hr,
            (32, 32),
            method="bicubic",
            antialias=True
        )

        return lr, label

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    def enhance(lr, labels):
        sr = generator(lr, training=False)
        sr = tf.clip_by_value(sr, -1.0, 1.0)
        return sr, labels

    ds = ds.map(enhance, num_parallel_calls=1)
    ds = ds.prefetch(AUTOTUNE)

    return ds
