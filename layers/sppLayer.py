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

class SPPBottleneck(tf.keras.layers.Layer):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        out_channels (int): The output channels of this Module.
        pool_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
    """
    def __init__(self, out_channels, pool_sizes=[5, 9, 13], **kwargs):
        super(SPPBottleneck, self).__init__(**kwargs)
        self.pool_sizes = pool_sizes
        self.conv_1 = tf.keras.layers.Conv2D(out_channels // 2, (1, 1), 1, padding="same", use_bias=False,
                                           kernel_initializer='lecun_normal',
                                          )
        self.batch_norm_1 = BatchNorm(momentum=0.03, epsilon=0.001)
        self.act_1 = tf.keras.layers.Activation('selu')

        self.conv_2 = tf.keras.layers.Conv2D(out_channels, (1, 1), 1, padding="same", use_bias=False,
                                           kernel_initializer='lecun_normal',
                                          )
        self.batch_norm_2 = BatchNorm(momentum=0.03, epsilon=0.001)
        self.act_2 = tf.keras.layers.Activation('selu')

    def call(self, x, training=True):
        # Initialize list to hold pooled features
        pooled_features = []

        x = self.conv_1(x)
        x = self.batch_norm_1(x, training)
        x = self.act_1(x)

        # Loop over each pool size
        for pool_size in self.pool_sizes:
            # Apply max pooling with the given pool size and stride size
            pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            # Append the pooled features to the list
            pooled_features.append(pool)

        # Concatenate the pooled features along the channel axis
        x = tf.concat(pooled_features, axis=-1)

        x = self.conv_2(x)
        x = self.batch_norm_2(x, training)
        x = self.act_2(x)

        return x

