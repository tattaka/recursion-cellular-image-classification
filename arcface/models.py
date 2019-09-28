from __future__ import division
import numpy as np
import re

import chainer

from chainer import static_code # WIP
from chainer import static_graph # WIP

from chainer import iterators
import chainer.functions as F
import chainer.links as L

import chainercv

from chainercv2.model_provider import get_model as chcv2_get_model

from functools import partial

class Face6chModel(chainer.Chain):

    def __init__(self, backborn, metric_fc, fc):
        super(Face6chModel, self).__init__()
        self.initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.backborn = backborn
            self.backborn.init_block = PreResInitBlock(in_channels = 6, out_channels=64)# Use Other
#             self.backborn.init_block = PreResInitBlock(in_channels = 6, out_channels=32)# Use efficientnetb1
            self.metric_fc = metric_fc
            self.fc = fc
            self.bn1 = L.BatchNormalization(axis=0)
            self.bn2 = L.BatchNormalization(axis=0)
            self.l1 = L.Linear(None, 1024 * 4, initialW=self.initializer)
            self.l2 = L.Linear(None, 512 * 4, initialW=self.initializer)
#     @static_graph
    def __call__(self, imgs):
        B = len(imgs)
        x = self.xp.array(imgs)
#         feature = self.backborn(F.cast(x, dtype=np.float16))
        feature = self.backborn(x)
        B = feature.shape[0]
        feature = F.reshape(feature, (B, -1))
        feature = self.bn1(feature)
        feature = F.dropout(feature, 0.25)
        feature = self.l1(feature)
        feature = F.relu(feature)
        feature = self.bn2(feature)
        feature = F.dropout(feature, 0.25)
        feature = self.l2(feature)
        feature = F.relu(feature)
        
        cosine = self.metric_fc(feature)
        out_fc = self.fc(feature)
        return cosine, out_fc
    
    def GlobalConcatPooling(self, x):
        return F.concat([F.average(x, axis=(2, 3), keepdims=True), F.mean(x, axis=(2, 3), keepdims=True)])
    
class PreResInitBlock(chainer.Chain):
    """
    PreResNet specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PreResInitBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=7,
                stride=2,
                pad=3,
                nobias=True)
            self.bn = L.BatchNormalization(size=out_channels)
            self.activ = F.relu
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x
        
