import tensorflow as tf

class BatchNorm(tf.keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def conv_bn_act(out_channels, kernel_size, strides, groups=1, name=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            BatchNorm(momentum=0.03, epsilon=0.001, name="bn"),
            tf.keras.layers.Activation('selu'),
        ], name=name
    )
    