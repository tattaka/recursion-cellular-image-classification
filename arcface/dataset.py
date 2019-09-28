import sys 
import numpy as np 
import pandas as pd 
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from PIL import Image
import cv2
import chainer 

from chainercv import utils
from chainercv import transforms

import copy

import math
import six
import random

DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)


class ImagesDataset(chainer.dataset.DatasetMixin):
    def __init__(self, df, img_dir, mode='train', channels=DEFAULT_CHANNELS):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    def _get_img_path(self, index):
        return  f'{self.img_dir}{self.mode}_rgb/'+self.records[index].filename
        
    def __len__(self):
        return self.len
    
    def get_example(self, i):
        img_path = self._get_img_path(i)
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        
        if self.mode == 'train':
            return img, self.records[i].sirna
        else:
            return img, self.records[i].id_code
        
# class ImagesDataset6ch(chainer.dataset.DatasetMixin):
#     def __init__(self, df, img_dir, mode='train', site=1, channels=DEFAULT_CHANNELS):
#         self.records = df.to_records(index=False)
#         self.channels = channels
#         self.site = site
#         self.mode = mode
#         self.img_dir = img_dir
#         self.len = df.shape[0]
        
#     def _load_img(self, file_name):
#         img = utils.read_image(file_name, dtype=np.float32, color=False)
#         return img

#     def _get_img_path(self, index, channel):
#         experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
#         return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        
#     def get_example(self, index):
#         paths = [self._get_img_path(index, ch) for ch in self.channels]
#         img = np.concatenate([self._load_img(img_path) for img_path in paths], axis=0)
#         if self.mode == 'train':
#             return img, int(self.records[index].sirna)
#         else:
#             return img, self.records[index].id_code

#     def __len__(self):
#         return self.len

class ImagesDataset6ch(chainer.dataset.DatasetMixin):
    def __init__(self, df, img_dir, mode='train'):
        self.records = df.to_records(index=False)
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    def _get_img_path(self, index):
        return  f'{self.img_dir}'+self.records[index].filename
        
    def __len__(self):
        return self.len
    
    def get_example(self, i):
        img_path = self._get_img_path(i)
        img = np.load(img_path).astype(np.float32)
        if self.mode == 'train':
            return img, self.records[i].sirna
        else:
            return img, self.records[i].id_code