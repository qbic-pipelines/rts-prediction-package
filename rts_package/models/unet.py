from typing import Any, Optional

import torch
from torch import nn

__all__ = ['UNET', 'NESTEDUNET', 'U2NET']

# Originally contained own implementation of was changed because of weight init
# and adopted from https://github.com/ShawnBIT/UNet-family/blob/master/networks/UNet.py

from rts_package.models.unet_super import UNetsuper
from rts_package.models.unet_utils import init_weights, unetConv2, unetUp, _size_map, _upsample_like, RSU


class UNET(UNetsuper):
    def __init__(self, num_classes, len_test_set, hparams, input_channels, min_filter, feature_scale=2, is_deconv=True,
                 is_batchnorm=True, **kwargs):
        super().__init__(num_classes, len_test_set, hparams, input_channels, min_filter, **kwargs)

        self.in_channels = input_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*512
        maxpool1 = self.maxpool(conv1)  # 16*256*256

        conv2 = self.conv2(maxpool1)  # 32*256*256
        maxpool2 = self.maxpool(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)  # 64*128*128
        maxpool3 = self.maxpool(conv3)  # 64*64*64

        conv4 = self.conv4(maxpool3)  # 128*64*64
        maxpool4 = self.maxpool(conv4)  # 128*32*32

        center = self.center(maxpool4)  # 256*32*32
        up4 = self.up_concat4(center, conv4)  # 128*64*64
        up3 = self.up_concat3(up4, conv3)  # 64*128*128
        up2 = self.up_concat2(up3, conv2)  # 32*256*256
        up1 = self.up_concat1(up2, conv1)  # 16*512*512

        final = self.final(up1)

        return torch.sigmoid(final)


class NESTEDUNET(UNetsuper):
    def __init__(self, num_classes, len_test_set, hparams, input_channels, min_filter, feature_scale=2, is_deconv=True,
                 is_batchnorm=True, is_ds=True, **kwargs):
        super().__init__(num_classes, len_test_set, hparams, input_channels, min_filter, **kwargs)
        self.in_channels = input_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return final
        else:
            return final_4


class U2NET(UNetsuper):
    def __init__(self, num_classes, len_test_set: int, input_channels=1, min_filter=32, **kwargs):
        super().__init__(num_classes, len_test_set, input_channels, min_filter, **kwargs)
        self._make_layers(input_channels, min_filter)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, input_channels, min_filter):
        cfgs = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, input_channels, min_filter, min_filter * 2), -1],
            'stage2': ['En_2', (6, min_filter * 2, min_filter, min_filter * 2 ** 2), -1],
            'stage3': ['En_3', (5, min_filter * 2 ** 2, min_filter * 2, min_filter * 2 ** 3), -1],
            'stage4': ['En_4', (4, min_filter * 2 ** 3, min_filter * 2 ** 2, min_filter * 2 ** 4), -1],
            'stage5': ['En_5', (4, min_filter * 2 ** 4, min_filter * 2 ** 3, min_filter * 2 ** 4, True), -1],
            'stage6': ['En_6', (4, min_filter * 2 ** 4, min_filter * 2 ** 3, min_filter * 2 ** 4, True),
                       min_filter * 2 ** 4],
            'stage5d': ['De_5', (4, min_filter * 2 ** 5, min_filter * 2 ** 3, min_filter * 2 ** 4, True),
                        min_filter * 2 ** 4],
            'stage4d': ['De_4', (4, min_filter * 2 ** 5, min_filter * 2 ** 2, min_filter * 2 ** 3),
                        min_filter * 2 ** 3],
            'stage3d': ['De_3', (5, min_filter * 2 ** 4, min_filter * 2, min_filter * 2 ** 2), min_filter * 2 ** 2],
            'stage2d': ['De_2', (6, min_filter * 2 ** 3, min_filter, min_filter * 2), min_filter * 2],
            'stage1d': ['De_1', (7, min_filter * 2 ** 2, int(min_filter * 2 ** (1 / 2)), min_filter * 2),
                        min_filter * 2],
        }
        cfgs = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, 1, 32, 64), -1],
            'stage2': ['En_2', (6, 64, 32, 128), -1],
            'stage3': ['En_3', (5, 128, 64, 256), -1],
            'stage4': ['En_4', (4, 256, 128, 512), -1],
            'stage5': ['En_5', (4, 512, 256, 512, True), -1],
            'stage6': ['En_6', (4, 512, 256, 512, True), 512],
            'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
            'stage4d': ['De_4', (4, 1024, 128, 256), 256],
            'stage3d': ['De_3', (5, 512, 64, 128), 128],
            'stage2d': ['De_2', (6, 256, 32, 64), 64],
            'stage1d': ['De_1', (7, 128, 16, 64), 64],
        }
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.num_classes, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.num_classes), self.num_classes, 1))

    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        loss = 0
        for logit in logits:
            loss += self.criterion(logit, labels)
        return loss

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        data, target = batch
        output = self.forward(data)
        _, prediction = torch.max(output[0], dim=1)
        return data, target, output, prediction
