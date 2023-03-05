import tensorflow as tf
from layers.cspLayer import CSPLayer
from typing import Sequence

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

class CSPNeXtPAFPN(tf.keras.layers.Layer):
    """Path Aggregation Network with CSPNeXt blocks.
    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
    """
    def __init__(   
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_factor: int=2,
        **kwargs
    ) -> None:
        super(CSPNeXtPAFPN, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = tf.keras.layers.UpSampling2D(upsample_factor)
        self.reduce_layers = []
        self.top_down_blocks = []

        for idx in range(len(in_channels)-1, 0, -1):
            self.reduce_layers.append(conv_bn_act(in_channels[idx - 1], 1, 1))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio))

        self.downsamples = []
        self.bottom_up_blocks = []
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(conv_bn_act(in_channels[idx], 3, 2))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio,))

        self.out_convs = []
        for i in range(len(in_channels)):
            self.out_convs.append(conv_bn_act(out_channels,3,1))

    def call(self, inputs, training=True):
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh, training=training)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                tf.concat([upsample_feat, feat_low], axis=-1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low, training=training)
            out = self.bottom_up_blocks[idx](
                tf.concat([downsample_feat, feat_height], axis=-1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx], training=training)

        return tuple(outs)
