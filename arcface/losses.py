import chainer
from chainer import cuda
from chainer.cuda import to_cpu
import chainer.functions as F
import chainer.links as L
import math
import numpy as np


class FocalLoss(object):
    def __init__(self, class_num=1108, gamma=2.0, alpha = 0.25, eps=1e-7, label_smoothing=False):
        self.gamma = gamma
        self.class_num = 1108
        self.alpha = alpha
        self.eps = eps
        self.ls = label_smoothing
        
    def loss(self, x, target):
        xp = chainer.cuda.get_array_module(target)
        logit = F.softmax(x)
        logit = F.clip(logit, x_min=self.eps, x_max=1-self.eps)
        if self.ls == False:
            loss_ce = F.softmax_cross_entropy(x, target)
        else:
            oh_target = xp.eye(self.class_num)[target]
            ls_target = self.label_smoothing(oh_target, epsilon=0.1, xp=xp)
            loss_ce = -F.sum(F.log_softmax(x) * ls_target) / ls_target.shape[0]
        loss_focal = loss_ce * self.alpha * (1 - logit) ** self.gamma
        return F.mean(loss_focal)
    
    def label_smoothing(self, target, xp, epsilon=0.1):
        a = xp.array(target)
        dim = target.shape[1]
        b = xp.ones(dim) * (1 / dim)
        return (1 - epsilon) * a + epsilon * b 
    
class AutoFocalLoss(object):
    def __init__(self, class_num=1108, h = 0.7, alpha = 0.25, eps=1e-7, label_smoothing=False):
        self.alpha = alpha
        self.class_num = 1108
        self.eps = eps
        self.h = h
        self.pc = 0
        self.ls = label_smoothing
        
    def loss(self, x, target):
        xp = chainer.cuda.get_array_module(target)
        logit = F.softmax(x)
        logit = F.clip(logit, x_min=self.eps, x_max=1-self.eps)
        if self.ls == False:
            loss_ce = F.softmax_cross_entropy(x, target)
        else:
            oh_target = xp.eye(self.class_num)[target]
            ls_target = self.label_smoothing(oh_target, epsilon=0.1, xp=xp)
            loss_ce = -F.sum(F.log_softmax(x) * ls_target) / ls_target.shape[0]
        
        self.pc = self.pc*0.95 + F.mean(logit, axis=0).data*0.05
        k = self.h * self.pc + (1 - self.h)
        gamma = F.log(1 - k) / F.log(1 - self.pc) - 1
        
        loss_focal = loss_ce * self.alpha * (1 - logit) ** gamma
        return F.mean(loss_focal)
    
    def label_smoothing(self, target, xp, epsilon=0.1):
        a = xp.array(target)
        dim = target.shape[1]
        b = xp.ones(dim) * (1 / dim)
        return (1 - epsilon) * a + epsilon * b 

class AutoFocalLossBCE(object):
    def __init__(self, class_num=1108, h = 0.7, alpha = 0.25, eps=1e-7, label_smoothing=False):
        self.alpha = alpha
        self.class_num = 1108
        self.eps = eps
        self.h = h
        self.pc = 0
        self.ls = label_smoothing
        
    def loss(self, x, target):
        xp = chainer.cuda.get_array_module(target)
        logit = F.softmax(x)
        logit = F.clip(logit, x_min=self.eps, x_max=1-self.eps)
        if self.ls == False:
            oh_target = xp.eye(self.class_num)[target]
            loss_ce = F.sigmoid_cross_entropy(x, oh_target)
        else:
            oh_target = xp.eye(self.class_num)[target]
            ls_target = self.label_smoothing(oh_target, epsilon=0.1, xp=xp)
            loss_ce = -F.sum(F.log(F.sigmoid(x)) * ls_target) / ls_target.shape[0]
        
        self.pc = self.pc*0.95 + F.mean(logit, axis=0).data*0.05
        k = self.h * self.pc + (1 - self.h)
        gamma = F.log(1 - k) / F.log(1 - self.pc) - 1
        
        loss_focal = loss_ce * self.alpha * (1 - logit) ** gamma
        return F.mean(loss_focal)
    
    def label_smoothing(self, target, xp, epsilon=0.1):
        a = xp.array(target)
        dim = target.shape[1]
        b = xp.ones(dim) * (1 / dim)
        return (1 - epsilon) * a + epsilon * b 

class ArcFace(object):
    def __init__(self, class_num=1108, s=30.0, m=0.50):
        self.class_num = class_num
        self.s = s
        self.m = m
        self.xp = None
        
    def loss(self, logits, labels):
        xp = chainer.cuda.get_array_module(logits)
        # add margin
        z = 1 - 1e-6
        theta = F.arccos(F.clip(logits, -z, z))
        target_logits = F.cos(theta + self.m)
        one_hot = xp.eye(self.class_num)[labels]
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s
        loss = F.softmax_cross_entropy(output, labels)
        return loss
    
class AdaCos(object):
    def __init__(self, class_num=1108, m=0.50):
        self.class_num = class_num
        self.s = math.sqrt(2) * math.log(self.class_num - 1)
        self.m = m
        self.xp = None
        
    def loss(self, logits, labels):
        xp = chainer.cuda.get_array_module(logits)
        # add margin
        z = 1 - 1e-6
        theta = F.arccos(F.clip(logits, -z, z))
        one_hot = xp.eye(self.class_num)[labels]
        
        
        B_avg = xp.where(one_hot < 1, xp.clip(xp.exp(self.s * logits.data), -z, z), xp.zeros_like(logits.data))
        B_avg = xp.sum(B_avg) / logits.shape[0]
        theta_med = xp.asarray(np.median(to_cpu(theta.data[one_hot == 1])))
        self.s = math.log(B_avg) / math.cos(min([math.pi/4, theta_med]))
        
        output = self.s * logits
        loss = F.softmax_cross_entropy(output, labels)
        return loss
    
    
import cupy

def median(x, axis):
    """配列(x)の軸(axis)に沿った中央値"""
    xp = cupy.get_array_module(x)
    n = x.shape[axis]
    s = xp.sort(x, axis)
    m_odd = xp.take(s, n // 2, axis)
    if n % 2 == 1:  # 奇数個
        return m_odd
    else:  # 偶数個のときは中間の値
        m_even = xp.take(s, n // 2 - 1, axis)
        return (m_odd + m_even) / 2