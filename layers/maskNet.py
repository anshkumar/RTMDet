import tensorflow as tf
from layers.convBnAct import conv_bn_act

class MaskFeatModule(tf.keras.layers.Layer):
    """Mask feature head used in RTMDet-Ins.
    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        stacked_convs (int): Number of convs in mask feature branch.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
        **kwargs) -> None:
        super(MaskFeatModule, self).__init__(**kwargs)
        self.num_levels = num_levels
        self.fusion_conv = tf.keras.layers.Conv2D(in_channels, 1, 1, padding="same",
                                                kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                              )
        convs = []
        for i in range(stacked_convs):
            convs.append(conv_bn_act(feat_channels, 3, 1, activation='relu'))

        self.stacked_convs = tf.keras.Sequential(convs)
        self.projection = tf.keras.layers.Conv2D(num_prototypes, 1)

    def call(self, features):
        # multi-level feature fusion
        fusion_feats = [features[0]]
        size = tf.shape(features[0])[1:-1]
        for i in range(1, self.num_levels):
            f = tf.image.resize(features[i], size=size)
            fusion_feats.append(f)

        fusion_feats = tf.concat(fusion_feats, axis=-1)
        fusion_feats = self.fusion_conv(fusion_feats)

        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)
        return mask_features
