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

_imagenet_mean = np.array([123.15163084, 115.90288257, 103.0626238], dtype=np.float32)[:, np.newaxis, np.newaxis]

class Config(object):
    def __init__(self):
        
        self.path_data = '../input/'
        self.train_df = "new_train_6ch.csv"
        self.test_df = "new_test_6ch.csv"
        self.input_shape = (256, 256)
        self.pseudo_labeling_path = None
        self.mean = 0
        self.dataset = ImagesDataset6ch
        self.train_transform = TrainTransform(img_size = self.input_shape)
        self.valid_trainsform = ValTransform(img_size = self.input_shape)
        self.valid_trainsform_flip1 = ValTransform(img_size = self.input_shape, x_flip=True) # for tta
        self.valid_trainsform_flip2 = ValTransform(img_size = self.input_shape, y_flip=True) # for tta
        self.valid_trainsform_flip3 = ValTransform(img_size = self.input_shape, x_flip = True, y_flip=True) # for tta

        self.communicator = 'pure_nccl'
        self.loaderjob = 4
        self.batchsize = 32
        self.out = 'result'
        self.device = 0


        self.num_class = 1108
        self.backborn_cfg = {'name':'seresnext50_32x4d', 'pretrain': True, 'layer': 'features'}
        
        self.fc_head = FullyConnection(self.num_class)
        self.metric_head = ArcMarginProduct(self.num_class)
        
        self.model = partial(Face6chModel, fc = self.fc_head, metric_fc=self.metric_head)
        
        self.fc_lossfun = 'auto_focal_loss'
        self.ls = True
        self.metric_lossfun = 'adacos'
        self.fix_sche = True
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
        
#         self.alpha = 5e-4 * (self.batchsize / 16)
#         self.optimizer = 'Adam'
        

        self.shift_lr = 0.1
        
        self.lr_points = [self.max_epoch*0.6, self.max_epoch*0.85]
        
        self.lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 5e-4
