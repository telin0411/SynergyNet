import torch
from torch import nn
from detr_dummy_args import dummy_args
from models.deformable_detr import build


__all__ = ['DETRNet', 'detrnet']


model_urls = {
    'detrnet': 'pretrained/r50_deformable_detr-checkpoint.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DETRNet(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(DETRNet, self).__init__()

        detr_args = dummy_args()
        detr_args.device = "cuda:0"
        self.detr, _, _ = build(detr_args)

        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10
        self.num_texture = 40
        self.num_bin = 121
        self.num_scale = 1
        self.num_trans = 3

        # self.last_channel = 256
        self.detr_proj = nn.Sequential(
            nn.Linear(256, 1280),
        )

        self.last_channel = 1280

        self.classifier_ori = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_ori),
        )

        self.classifier_shape = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_shape),
        )
        self.classifier_exp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_exp),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        # x = self.features(x)
        x = self.detr(x)
        x = x["hs"][0, :, 0, :]
        x = self.detr_proj(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]

        # x = nn.functional.adaptive_avg_pool2d(x, 1)
        # x = x.reshape(x.shape[0], -1)

        pool_x = x.clone()

        ## Bin-based
        # pre_yaw = self.classifier_yaw(x)
        # pre_pitch = self.classifier_pitch(x)
        # pre_roll = self.classifier_roll(x)
        # pre_scale = self.classifier_scale(x)
        # pre_trans = self.classifier_trans(x)

        x_ori = self.classifier_ori(x)

        #GeoNet
        # x_geo = self.geoNet(x.unsqueeze(2)).squeeze(2)
        # x_shape = self.classifier_shape(x_geo)
        # x_exp = self.classifier_exp(x_geo)

        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)

        #x_tex = self.classifier_texture(x)
        #x = torch.cat((x_ori, x_shape, x_exp, x_tex), dim=1)
        x = torch.cat((x_ori, x_shape, x_exp), dim=1)
        
        #x = torch.cat((pre_yaw, pre_pitch, pre_roll, pre_scale, pre_trans, x_shape, x_exp, x_tex), dim=1)
        
        return x, pool_x

    def forward(self, x):
        return self._forward_impl(x)


def detrnet(pretrained=False, progress=True, **kwargs):
    """
    Constructs a architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = DETRNet(**kwargs)
    checkpoint = torch.load(model_urls["detrnet"], map_location='cpu')
    model.detr.load_state_dict(checkpoint['model'], strict=True)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model
