import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_srgan_dataset(csv_path, batch_size=16, training=False):
    df = pd.read_csv(csv_path)
    paths = df["filepath"].astype(str).tolist()

    ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    def preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(
            image, (128, 128),
            method="bicubic",
            antialias=True
        )

        if training:
            image = tf.image.random_flip_left_right(image)

        # High-resolution image: [-1, 1]
        hr = tf.clip_by_value(image / 127.5 - 1.0, -1.0, 1.0)

        # Low-resolution image: 32x32, also [-1, 1]
        lr = tf.image.resize(
            hr, (32, 32),
            method="bicubic",
            antialias=True
        )
        lr = tf.clip_by_value(lr, -1.0, 1.0)

        return lr, hr

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds
