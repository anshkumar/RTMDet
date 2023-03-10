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

class WeightStandardizedConv2D(tf.keras.layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="SAME",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

class WeightStandardizedSepConv2D(tf.keras.layers.SeparableConv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.depthwise_conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="SAME",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

def conv_bn_act(out_channels, kernel_size, strides, groups=1, activation='selu', name=None):
    return tf.keras.Sequential(
        [
            WeightStandardizedConv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            BatchNorm(momentum=0.03, epsilon=0.001, name="bn"),
            tf.keras.layers.Activation(activation),
        ], name=name
    )

def sep_conv_bn_act(out_channels, kernel_size, strides, groups=1, activation='selu', name=None):
    return tf.keras.Sequential(
        [
            WeightStandardizedSepConv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            BatchNorm(momentum=0.03, epsilon=0.001, name="bn"),
            tf.keras.layers.Activation(activation),
        ], name=name
    )
