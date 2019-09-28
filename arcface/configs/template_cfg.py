import sys
from functools import partial

from chainer.training.triggers import ManualScheduleTrigger
from chainer.optimizers import CorrectedMomentumSGD, Adam, RMSprop
from chainer.training import extensions
from chainer import functions as F

from dataset import *
from metrics import *
from losses import * 
from utils import * 
from updater import * 
from evaluator import * 
from models import * 


class Config(object):
    def __init__(self):
        
        self.path_data = '../input/'
        self.train_df = "new_train_6ch.csv"
        self.test_df = "new_test_6ch.csv"
        self.pseudo_labeling = True
        self.input_shape = (256, 256)
        self.mean = 0
        self.dataset = ImagesDataset6ch
        self.train_transform = TrainTransform(img_size = self.input_shape)
        self.valid_trainsform = ValTransform(img_size = self.input_shape)
        self.valid_trainsform_flip1 = ValTransform(img_size = self.input_shape, x_flip=True) # for tta
        self.valid_trainsform_flip2 = ValTransform(img_size = self.input_shape, y_flip=True) # for tta
        self.valid_trainsform_flip3 = ValTransform(img_size = self.input_shape, x_flip = True, y_flip=True) # for tta

        self.communicator = 'pure_nccl'
        self.loaderjob = 3
        self.batchsize = 32
        self.out = 'result'
        self.device = 0


        self.num_class = 1108
#         self.backborn_lists = [
#             {'name': 'seresnext50_32x4d', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'seresnext101_32x4d', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'densenet121', 'pretrain': True, 'layer': 'features'} # work
#             {'name':'resnet50', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'resnet101', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'resnet152', 'pretrain': True, 'layer': 'features'},
#             {'name':'efficientnet_b0', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b1', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b2', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b3', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b4', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b5', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b6', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b7', 'pretrain': False, 'layer': 'features'},
#             {'name':'efficientnet_b0b', 'pretrain': True, 'layer': 'features'},
#             {'name':'efficientnet_b1b', 'pretrain': True, 'layer': 'features'}, # not work
#             {'name':'efficientnet_b2b', 'pretrain': True, 'layer': 'features'},
#             {'name':'efficientnet_b3b', 'pretrain': True, 'layer': 'features'},
#             {'name':'inceptionv4', 'pretrain': True, 'layer': 'features'},
#             {'name':'resnetd50b', 'pretrain': True, 'layer': 'features'},
#             {'name': 'airnext50_32x4d_r2', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'densenet169', 'pretrain': True, 'layer': 'features'},  
#             {'name':'densenet201', 'pretrain': True, 'layer': 'features'}, 
#             {'name':'bn_vgg16', 'pretrain': True, 'layer': 'features'}, 
#]
        
        self.backborn_cfg = {'name':'resnet50', 'pretrain': True, 'layer': 'features'}
        
        if self.backborn_cfg['name'] == 'efficientnet_b0' or self.backborn_cfg['name'] == 'efficientnet_b0b':
            self.input_shape = (224, 224)
        elif self.backborn_cfg['name'] == 'efficientnet_b1' or self.backborn_cfg['name'] == 'efficientnet_b1b':
            self.input_shape = (240, 240)
        elif self.backborn_cfg['name'] == 'efficientnet_b2' or self.backborn_cfg['name'] == 'efficientnet_b2b':
            self.input_shape = (260, 260)
        elif self.backborn_cfg['name'] == 'efficientnet_b3' or self.backborn_cfg['name'] == 'efficientnet_b3b':
            self.input_shape = (300, 300)
        elif self.backborn_cfg['name'] == 'efficientnet_b4':
            self.input_shape = (380, 380)
        elif self.backborn_cfg['name'] == 'efficientnet_b5':
            self.input_shape = (456, 456)
        elif self.backborn_cfg['name'] == 'efficientnet_b6':
            self.input_shape = (528, 528)
        elif self.backborn_cfg['name'] == 'efficientnet_b7':
            self.input_shape = (600, 600)
        
        self.fc_head = FullyConnection(self.num_class)
        self.metric_head = ArcMarginProduct(self.num_class)
        
        self.model = partial(Face6chModel, fc = self.fc_head, metric_fc=self.metric_head)
        
#         self.fc_lossfun = 'focal_loss'
        self.fc_lossfun = 'auto_focal_loss'
        self.ls = True
#         self.fc_lossfun = 'softmax_cross_entropy'
        self.metric_lossfun = 'adacos'
#         self.metric_lossfun = 'arcface'
        self.metric_w = 1
        self.fc_w = 0.1
        
        self.out = self.out +'_'+ str(self.backborn_cfg['name'])

        self.updater = MyFaceUpdater
        self.evaluator = MyFaceEvaluator


        self.fold = 4
        self.fold_target = 'cell'
        self.max_epoch = 90 
        
        self.lr = 2e-2 * (self.batchsize / 16) # initial learning rate
        self.optimizer = 'CorrectedMomentumSGD'
#         self.optimizer = 'NesterovAG'
        
        self.alpha = 2e-3 * (self.batchsize / 16)
#         self.optimizer = 'Adam'
        

        self.shift_lr = 0.1
        
        self.lr_points = [self.max_epoch*0.5, self.max_epoch*0.75]
        
        self.lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 5e-4
        