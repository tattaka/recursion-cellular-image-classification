import six
import copy
import numpy as np

import chainer
from chainer import Variable
from chainer import backend
from chainer.backends import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import _updater
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import links as L
from chainer import functions as F
from chainer import reporter

from functools import partial

from losses import *
    
class MyFaceUpdater(chainer.training.StandardUpdater):
    def __init__(self, iterator, opt, net, metric_lossfun, fc_lossfun, max_epoch, fix_sche = False, metric_w = 1, fc_w = 1, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self._optimizers = {"main":opt}
        self.net = net
        self.fc_lossfun = fc_lossfun
        self.metric_lossfun = metric_lossfun
        self.fc_w = fc_w
        self.metric_w = metric_w
        self.max_epoch = max_epoch # WIP
        self.converter = convert.concat_examples
        self._device = device
        self.fix_sche = fix_sche
        self.iteration = 0

    def update_core(self):
        """lossを計算"""
        iterator = self._iterators['main'].next()

        xp = np if int(self._device) == -1 else cuda.cupy
        input = self.converter(iterator, self._device, padding=0)
        x_batch = xp.asarray(input[0], dtype=xp.float32)
        t_batch = xp.asarray(input[1],  dtype=xp.int32)
        
        self.loss = None
        self.acc = None
        
        cosine, logits_fc = self.net(x_batch)
        if self.fix_sche:
            metric_w = self.metric_w - (self.metric_w - 0.1) / int(self.max_epoch / 10) * int(self.epoch / 10)
            fc_w = self.fc_w + (1 - self.fc_w) / int(self.max_epoch / 10) * int(self.epoch / 10)
        else:
            metric_w = self.metric_w
            fc_w = self.fc_w
        
        metric_loss = self.metric_lossfun(cosine, t_batch)
        fc_loss = self.fc_lossfun(logits_fc, t_batch)
        self.loss = metric_loss * metric_w / (metric_w + fc_w) + fc_loss * fc_w / (metric_w + fc_w)
        self.acc = F.accuracy(logits_fc, t_batch)
        
        self._optimizers["main"].target.cleargrads()
        self.loss.backward()
        self._optimizers["main"].update()
        reporter.report({'main/loss':self.loss,'main/face_loss':metric_loss, 'main/ce_loss':fc_loss, 'main/accuracy':self.acc})