from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf

from srgan_models import PixelShuffle

tf.keras.utils.set_random_seed(42)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

generator = tf.keras.models.load_model(
    "checkpoints/srgan/best_gan_generator.keras",
    custom_objects={"PixelShuffle": PixelShuffle},
    compile=False
)

output_root = Path("data/srgan_enhanced")
output_root.mkdir(parents=True, exist_ok=True)

splits = {
    "train_final": "data/splits/train_final.csv",
    "val": "data/splits/val.csv",
    "test": "data/splits/test.csv",
}

BATCH_SIZE = 16

for split_name, csv_path in splits.items():
    df = pd.read_csv(csv_path)

    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []

    print(f"\nProcessing {split_name}: {len(df)} images", flush=True)

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        lr_images = []

        for path in batch["filepath"]:
            image = tf.io.read_file(str(path))
            image = tf.image.decode_jpeg(image, channels=3)

            hr = tf.image.resize(
                image, (128, 128),
                method="bicubic",
                antialias=True
            )
            hr = tf.cast(hr, tf.float32) / 127.5 - 1.0

            lr = tf.image.resize(
                hr, (32, 32),
                method="bicubic",
                antialias=True
            )

            lr_images.append(lr)

        lr_batch = tf.stack(lr_images)
        sr_batch = generator(lr_batch, training=False)

        sr_batch = tf.clip_by_value(sr_batch, -1.0, 1.0)
        sr_batch = tf.cast(
            tf.round((sr_batch + 1.0) * 127.5),
            tf.uint8
        )

        for i, (_, row) in enumerate(batch.iterrows()):
            original_name = Path(row["filepath"]).stem
            output_path = split_dir / f"{original_name}.png"

            png = tf.io.encode_png(sr_batch[i])
            tf.io.write_file(str(output_path), png)

            output_rows.append({
                "filepath": str(output_path),
                "label": int(row["label"])
            })

        processed = min(start + BATCH_SIZE, len(df))

        if processed % 512 < BATCH_SIZE or processed == len(df):
            print(
                f"{split_name}: {processed}/{len(df)}",
                flush=True
            )

    output_csv = Path("data/splits") / f"{split_name}_srgan.csv"
    pd.DataFrame(output_rows).to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}", flush=True)

print("\nAll SRGAN-enhanced datasets generated successfully.")
