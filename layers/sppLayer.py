import tensorflow as tf
from layers.convBnAct import conv_bn_act

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
        self.conv_1 = conv_bn_act(out_channels // 2, (1, 1), 1)
        self.conv_2 = conv_bn_act(out_channels, (1, 1), 1,)

    def call(self, x, training=True):
        # Initialize list to hold pooled features
        pooled_features = []

        x = self.conv_1(x, training=training)

        # Loop over each pool size
        for pool_size in self.pool_sizes:
            # Apply max pooling with the given pool size and stride size
            pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            # Append the pooled features to the list
            pooled_features.append(pool)

        # Concatenate the pooled features along the channel axis
        x = tf.concat(pooled_features, axis=-1)

        x = self.conv_2(x, training=training)

        return x

