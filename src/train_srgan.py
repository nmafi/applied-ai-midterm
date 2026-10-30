import os
import sys
import csv
import argparse
from pathlib import Path

import tensorflow as tf

from srgan_models import build_discriminator, PixelShuffle
from srgan_data import load_srgan_dataset

tf.keras.utils.set_random_seed(42)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

out = Path("checkpoints/srgan")
results = Path("results/srgan")
out.mkdir(parents=True, exist_ok=True)
results.mkdir(parents=True, exist_ok=True)

train_ds = load_srgan_dataset(
    "data/splits/train_final.csv",
    batch_size=args.batch_size,
    training=True
)
val_ds = load_srgan_dataset(
    "data/splits/val.csv",
    batch_size=args.batch_size,
    training=False
)

generator = tf.keras.models.load_model(
    out / "best_pretrained_fixed.keras",
    compile=False
)
discriminator = build_discriminator()

vgg = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)
vgg.trainable = False

feature_extractor = tf.keras.Model(
    vgg.input,
    vgg.get_layer("block5_conv4").output
)
feature_extractor.trainable = False

g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9)
d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9)

bce = tf.keras.losses.BinaryCrossentropy()

def vgg_features(images):
    images = (images + 1.0) * 127.5
    images = tf.keras.applications.vgg19.preprocess_input(images)
    return feature_extractor(images, training=False)

@tf.function
def train_step(lr, hr):
    with tf.GradientTape() as d_tape:
        fake = generator(lr, training=True)
        real_pred = discriminator(hr, training=True)
        fake_pred = discriminator(tf.stop_gradient(fake), training=True)

        d_loss = (
            bce(tf.ones_like(real_pred), real_pred)
            + bce(tf.zeros_like(fake_pred), fake_pred)
        ) / 2.0

    d_grads = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )
    d_optimizer.apply_gradients(
        zip(d_grads, discriminator.trainable_variables)
    )

    with tf.GradientTape() as g_tape:
        fake = generator(lr, training=True)
        fake_pred = discriminator(fake, training=False)

        pixel_loss = tf.reduce_mean(tf.square(hr - fake))

        real_features = tf.stop_gradient(vgg_features(hr))
        fake_features = vgg_features(fake)

        perceptual_loss = tf.reduce_mean(
            tf.square(real_features - fake_features)
        )

        adversarial_loss = bce(
            tf.ones_like(fake_pred), fake_pred
        )

        g_loss = (
            pixel_loss
            + 0.006 * perceptual_loss
            + 0.001 * adversarial_loss
        )

    g_grads = g_tape.gradient(
        g_loss, generator.trainable_variables
    )
    g_optimizer.apply_gradients(
        zip(g_grads, generator.trainable_variables)
    )

    return g_loss, d_loss, pixel_loss, perceptual_loss, adversarial_loss


@tf.function
def validation_step(lr, hr):
    fake = generator(lr, training=False)

    mse = tf.reduce_mean(tf.square(hr - fake))
    hr01 = tf.clip_by_value((hr + 1.0) / 2.0, 0.0, 1.0)
    fake01 = tf.clip_by_value((fake + 1.0) / 2.0, 0.0, 1.0)

    psnr = tf.reduce_mean(
        tf.image.psnr(hr01, fake01, max_val=1.0)
    )
    ssim = tf.reduce_mean(
        tf.image.ssim(hr01, fake01, max_val=1.0)
    )

    return mse, psnr, ssim


history_path = results / "gan_history.csv"

epoch_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
best_mse_var = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)

checkpoint = tf.train.Checkpoint(
    epoch=epoch_counter,
    best_mse=best_mse_var,
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer
)

manager = tf.train.CheckpointManager(
    checkpoint,
    directory=str(out / "training_state"),
    max_to_keep=3
)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("Resumed from:", manager.latest_checkpoint, flush=True)
else:
    print("Starting new adversarial training.", flush=True)

if not history_path.exists() or int(epoch_counter.numpy()) == 0:
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "g_loss", "d_loss", "pixel_loss",
            "perceptual_loss", "adversarial_loss",
            "val_mse", "val_psnr", "val_ssim"
        ])

best_val_mse = float(best_mse_var.numpy())

for epoch in range(int(epoch_counter.numpy()), args.epochs):
    metrics = [tf.keras.metrics.Mean() for _ in range(5)]

    for step, (lr, hr) in enumerate(train_ds):
        if args.steps is not None and step >= args.steps:
            break

        values = train_step(lr, hr)
        for metric, value in zip(metrics, values):
            metric.update_state(value)

    val_metrics = [tf.keras.metrics.Mean() for _ in range(3)]

    for lr, hr in val_ds:
        values = validation_step(lr, hr)
        for metric, value in zip(val_metrics, values):
            metric.update_state(value)

    train_values = [float(m.result()) for m in metrics]
    val_values = [float(m.result()) for m in val_metrics]

    print(
        f"Epoch {epoch+1}/{args.epochs} | "
        f"G: {train_values[0]:.4f} | "
        f"D: {train_values[1]:.4f} | "
        f"Val MSE: {val_values[0]:.5f} | "
        f"PSNR: {val_values[1]:.2f} | "
        f"SSIM: {val_values[2]:.4f}",
        flush=True
    )

    with open(history_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch + 1] + train_values + val_values
        )

    if val_values[0] < best_val_mse:
        best_val_mse = val_values[0]
        generator.save(out / "best_gan_generator.keras")

    epoch_counter.assign(epoch + 1)
    best_mse_var.assign(best_val_mse)

    if (epoch + 1) % 5 == 0:
        manager.save(checkpoint_number=epoch + 1)

    if (epoch + 1) % 10 == 0:
        generator.save(
            out / f"generator_epoch_{epoch+1:03d}.keras"
        )
        discriminator.save(
            out / f"discriminator_epoch_{epoch+1:03d}.keras"
        )

manager.save(checkpoint_number=int(epoch_counter.numpy()))

generator.save(out / "final_gan_generator.keras")
discriminator.save(out / "final_gan_discriminator.keras")

print("\nTraining completed.", flush=True)
