import tensorflow as tf
from tensorflow.keras import layers, Model



@tf.keras.utils.register_keras_serializable(package="SRGAN")
class PixelShuffle(layers.Layer):
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] * self.scale,
            input_shape[2] * self.scale,
            input_shape[3] // (self.scale ** 2)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def residual_block(x, filters=64):
    skip = x

    y = layers.Conv2D(
        filters, 3, padding="same"
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.PReLU(shared_axes=[1, 2])(y)

    y = layers.Conv2D(
        filters, 3, padding="same"
    )(y)
    y = layers.BatchNormalization()(y)

    return layers.Add()([skip, y])


def upsample_block(x, filters=256):
    x = layers.Conv2D(
        filters, 3, padding="same"
    )(x)

    # Pixel shuffle: 2x spatial upsampling
    x = PixelShuffle(scale=2)(x)

    return layers.PReLU(shared_axes=[1, 2])(x)


def build_generator(num_residual_blocks=8):
    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(
        64, 9, padding="same"
    )(inputs)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    skip = x

    for _ in range(num_residual_blocks):
        x = residual_block(x)

    x = layers.Conv2D(
        64, 3, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, skip])

    x = upsample_block(x)
    x = upsample_block(x)

    outputs = layers.Conv2D(
        3, 9, padding="same", activation="tanh"
    )(x)

    return Model(inputs, outputs, name="SRGAN_Generator")


if __name__ == "__main__":
    generator = build_generator()
    generator.summary()

    dummy = tf.random.uniform(
        (1, 32, 32, 3),
        minval=-1,
        maxval=1
    )

    output = generator(dummy)

    print("\nInput shape:", dummy.shape)
    print("Output shape:", output.shape)
    print("Output range:",
          float(tf.reduce_min(output)),
          float(tf.reduce_max(output)))


def discriminator_block(x, filters, stride=1, use_bn=True):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same"
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    return layers.LeakyReLU(negative_slope=0.2)(x)


def build_discriminator():
    inputs = layers.Input(shape=(128, 128, 3))

    x = discriminator_block(
        inputs, 64, stride=1, use_bn=False
    )
    x = discriminator_block(x, 64, stride=2)

    x = discriminator_block(x, 128, stride=1)
    x = discriminator_block(x, 128, stride=2)

    x = discriminator_block(x, 256, stride=1)
    x = discriminator_block(x, 256, stride=2)

    x = discriminator_block(x, 512, stride=1)
    x = discriminator_block(x, 512, stride=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(
        inputs,
        outputs,
        name="SRGAN_Discriminator"
    )
