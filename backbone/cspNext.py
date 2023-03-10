import tensorflow as tf
from layers.cspLayer import CSPLayer
from layers.sppLayer import SPPBottleneck
from typing import Sequence, Tuple
from layers.convBnAct import conv_bn_act

class CSPNeXt(tf.keras.layers.Layer):
    """CSPNeXt backbone used in RTMDet.
    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
    """
    # From left to right:
    # out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[128, 3, True, False], [256, 6, True, False],
               [512, 6, True, False], [1024, 3, False, True]],
        'P6': [[128, 3, True, False], [256, 6, True, False],
               [512, 6, True, False], [768, 3, True, False],
               [1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        **kwargs
        # init_cfg: OptMultiConfig = dict(
        #     type='Kaiming',
        #     layer='Conv2d',
        #     a=math.sqrt(5),
        #     distribution='uniform',
        #     mode='fan_in',
        #     nonlinearity='leaky_relu')
    ) -> None:
        super(CSPNeXt, self).__init__(**kwargs)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        self.out_indices = out_indices

        self.layers = []
        for i, (out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            stage.append(conv_bn_act(out_channels=out_channels, kernel_size=3, strides=2,))
            if use_spp:
                stage.append(SPPBottleneck(out_channels=out_channels, pool_sizes=spp_kernel_sizes,))
            stage.append(CSPLayer(out_channels=out_channels, num_blocks=num_blocks, add_identity=add_identity, expand_ratio=expand_ratio, channel_attention=channel_attention, ))
            self.layers.append(tf.keras.Sequential(stage, name='stage'+str(i)))

    def call(self, x, training=True):
        outs = []

        for i, layer in enumerate(self.layers):
            x = layer(x, training=training)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
