import tensorflow as tf
import tensorflow_addons as tfa
from layers.convBnAct import conv_bn_act, sep_conv_bn_act

class CSPNeXtBlock(tf.keras.layers.Layer):
    """The basic bottleneck block used in CSPNeXt.

    Args:
        out_channels (int): The output channels of this Module.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
    """
    def __init__(self, out_channels, expansion=0.5, add_identity=True, **kwargs):
        super(CSPNeXtBlock, self).__init__(**kwargs)
        self.conv_1 = conv_bn_act(int(out_channels * expansion), (3, 3), 1)
        self.conv_2 = sep_conv_bn_act(out_channels, (5, 5), 1)
        self.add_identity = add_identity

    def call(self, x, training=True):
        identity = x

        x = self.conv_1(x, training=training)
        x = self.conv_2(x, training=training)

        if self.add_identity:
            x = x + identity

        return x

class ChannelAttention(tf.keras.layers.Layer):
    """Channel attention Module.
    Args:
        channels (int): The input (and output) channels of the attention layer.
    """

    def __init__(self, channels, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.global_avgpool = tfa.layers.AdaptiveAveragePooling2D(1)
        self.fc = tf.keras.layers.Conv2D(channels, 1, 1, padding="same")

    def forward(self, x):
        """Forward function for ChannelAttention."""
        # TODO: Disable Mixed precision for AdaptiveAveragePooling2D
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = tf.keras.activations.hard_sigmoid(out)
        return x * out

class CSPLayer(tf.keras.layers.Layer):
    """Cross Stage Partial Layer.

    Args:
        out_channels (int): The output channels of this Module.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
    """
    def __init__(self, out_channels, expand_ratio=0.5, num_blocks=1, add_identity=True, channel_attention=False, **kwargs):
        super(CSPLayer, self).__init__(**kwargs)
        self.channel_attention = channel_attention

        mid_channels = int(out_channels * expand_ratio)
        self.short_conv = sep_conv_bn_act(mid_channels, (1, 1), 1)
        self.main_conv = conv_bn_act(mid_channels, (1, 1), 1)
        self.blocks = tf.keras.Sequential([CSPNeXtBlock(mid_channels, 1.0, add_identity) for _ in range(num_blocks)])
        self.final_conv = sep_conv_bn_act(out_channels, (1, 1), 1)

        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def call(self, x, training=True):
        x_short = self.short_conv(x, training=training)
        x_main = self.main_conv(x, training=training)
        x_main = self.blocks(x_main, training=training)

        x_final = tf.concat((x_main, x_short), axis=-1)

        if self.channel_attention:
            x_final = self.attention(x_final)
            
        x_final = self.final_conv(x_final, training=training)
        return x_final
