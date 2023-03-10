import tensorflow as tf
from layers.convBnAct import conv_bn_act

class RTMDetInsSepBNHead(tf.keras.layers.Layer):
    """Detection Head of RTMDet-Ins with sep-bn layers.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_feats (int): Number of features used for predictions.
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        stacked_convs (int): Number of stacked convs layers of prediction.
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        mask_loss_stride (int): Down sample stride of the masks for loss
            computation. Defaults to 4.

    """
    def __init__(self,
                 num_classes: int,
                 num_feats: int,
                 pred_kernel_size: int = 1,
                 stacked_convs: int = 2,
                 num_prototypes: int = 8,
                 dyconv_channels: int = 8,
                 num_dyconvs: int = 3,
                 mask_loss_stride: int = 4,
                 **kwargs) -> None:
        super(RTMDetInsSepBNHead, self).__init__()
        self.share_conv = share_conv
        self.stacked_convs = stacked_convs
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.mask_loss_stride = mask_loss_stride

        """Initialize layers of the head."""
        self.kernel_convs = []

        self.rtm_cls = []
        self.rtm_reg = []
        self.rtm_kernel = []
        self.rtm_obj = []

         # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        cls_convs = []
        reg_convs = []
        
        for i in range(self.stacked_convs):
            cls_convs.append(conv_bn_act(self.feat_channels, 3, 1))
            reg_convs.append(conv_bn_act(self.feat_channels, 3, 1))
            

        self.cls_convs = tf.keras.Sequential(cls_convs)
        self.reg_convs = tf.keras.Sequential(reg_convs)

        for _ in range(num_feats):
            kernel_convs = []
            for i in range(self.stacked_convs):
                kernel_convs.append(conv_bn_act(self.feat_channels, 3, 1))

            self.kernel_convs.append(tf.keras.Sequential(kernel_convs))
            self.rtm_cls.append(tf.keras.layers.Conv2D(self.num_class * self.num_anchors, self.pred_kernel_size, 1, padding="same",
                                                kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                              ))
            self.rtm_reg.append(tf.keras.layers.Conv2D(4 * self.num_anchors, self.pred_kernel_size, 1, padding="same",
                                              kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                            ))
            self.rtm_kernel.append(tf.keras.layers.Conv2D(self.num_gen_params, self.pred_kernel_size, 1, padding="same",
                                              kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.03),
                                            ))

    def call(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)
        cls_scores = []
        bbox_preds = []
        kernel_preds = []

        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            cls_feat = self.cls_convs(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            kernel_feat = self.kernel_convs[idx](kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            reg_feat = self.reg_convs(reg_feat)

            reg_dist = tf.keras.activations.relu(self.rtm_reg[idx](reg_feat)) #* stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)

        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat
        