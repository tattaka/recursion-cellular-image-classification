import chainer
from chainercv import transforms
import numpy as np
from chainer import functions as F
import random
from albumentations import RandomBrightnessContrast, GaussNoise

class TrainTransform(object):

    def __init__(self, img_size=(512, 512)):
        self.size = img_size

    def __call__(self, in_data):     
        img, label= in_data
#         img = transforms.resize(img, self.size)
        img = transforms.random_crop(img, self.size)
        img = transforms.random_rotate(img) 
        img = transforms.random_flip(img, x_random=True, y_random=True, return_param=False)
#         aug = RandomBrightnessContrast(p=0.5) # WIP
# #         aug = GaussNoise(p=0.5) # WIP
#         img = aug(image=img)['image']
######################## perimage normalize #############
        ms_im = img.reshape((img.shape[0], -1))
        mean = np.mean(ms_im, axis = 1)[:, np.newaxis, np.newaxis]
        std = np.std(ms_im, axis = 1)[:, np.newaxis, np.newaxis]
        img = (img - mean) / (std + 0.0000001)
####################################################
        return img, label
    
class ValTransform(object):

    def __init__(self, img_size=(512, 512), y_flip=False, x_flip=False):
        self.size = img_size
        self.y_flip = y_flip
        self.x_flip = x_flip

    def __call__(self, in_data):
        img, label = in_data
#         img = transforms.resize(img, self.size)
        img = transforms.random_crop(img, self.size)
        img = transforms.flip(img, y_flip=self.y_flip, x_flip=self.x_flip)
        img = transforms.random_rotate(img)
######################## perimage normalize #############
        ms_im = img.reshape((img.shape[0], -1))
        mean = np.mean(ms_im, axis = 1)[:, np.newaxis, np.newaxis]
        std = np.std(ms_im, axis = 1)[:, np.newaxis, np.newaxis]
        img = (img - mean) / (std + 0.0000001)
####################################################
        return img, label