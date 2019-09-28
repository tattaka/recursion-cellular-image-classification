import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import math
from tqdm import tqdm_notebook as tqdm

import chainer
from chainer import cuda
from chainer.cuda import to_cpu
import chainer.functions as F
import chainer.links as L
    
class FullyConnection(chainer.Chain):
    def __init__(self, out_features):
        self.initializer = chainer.initializers.HeNormal()
        self.out_features = out_features
        
        super(FullyConnection, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, out_features, initialW=self.initializer)
            self.bn1 = L.BatchNormalization(axis=0)
    def forward(self, input):
        xp =  input.xp
        output = self.l1(F.dropout(self.bn1(input), 0.25))
        return output
    
class ArcMarginProduct(chainer.Chain):
    def __init__(self, out_features):
        self.initializer = chainer.initializers.HeNormal()
        self.out_features = out_features
        
        super(ArcMarginProduct, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, out_features, initialW=self.initializer)
            
    def forward(self, input):
        xp =  input.xp
        cosine = self.l1(F.normalize(input))
        
        return cosine