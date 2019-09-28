import six
import copy
import numpy as np

import chainer
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


class MyFaceEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, net, metric_lossfun, fc_lossfun, max_epoch, fix_sche = False, converter=convert.concat_examples, metric_w = 1, fc_w = 1, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'main':net}
        self.converter = converter
        self._device = device
        self.fc_lossfun = fc_lossfun
        self.metric_lossfun = metric_lossfun
        self.fc_w = fc_w
        self.metric_w = metric_w
        self.max_epoch = max_epoch # WIP
        self.epoch = 0 #WIP
        self.xp = np if int(self._device) == -1 else cuda.cupy
        self.fix_sche = fix_sche

    def evaluate(self):
        iterator = self._iterators['main']
        self.net = self._targets['main']
        iterator.reset()
        it = iterator
#         it = copy.copy(iterator)
        summary = reporter.DictSummary()
    
        self.epoch = self.epoch + int(self.max_epoch / 10)
        
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                input = self.converter(batch, self._device, padding=0)
                x_batch = self.xp.asarray(input[0], dtype=self.xp.float32)
                t_batch = self.xp.asarray(input[1],  dtype=self.xp.int32)
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
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
                
                observation['validation/main/loss'] = self.loss
                observation['validation/main/face_loss'] = metric_loss
                observation['validation/main/ce_loss'] = fc_loss
                observation['validation/main/accuracy'] = self.acc
            
            summary.add(observation)
        return summary.compute_mean()