from __future__ import division
import argparse
import multiprocessing
import numpy as np
import re
import sys
from sklearn.model_selection import StratifiedKFold
import time

import copy

import chainer

from chainer.optimizer import WeightDecay
from chainer.optimizers import CorrectedMomentumSGD, Adam, NesterovAG
from chainer import training
from chainer.training import extensions

chainer.config.autotune = True
# chainer.config.cudnn_fast_batch_normalization = True
# chainer.global_config.dtype = np.dtype('float16') # WIP
chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset

from chainercv.chainer_experimental.training.extensions import make_shift
from chainer.training.triggers import ManualScheduleTrigger

import chainercv

import chainermn

from chainercv2.model_provider import get_model as chcv2_get_model

from functools import partial

from losses import * 

import pandas as pd
# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

import argparse 
from importlib import import_module


def stratified_groups_kfold(df, target, n_splits=5, random_state=0):
    all_groups = pd.Series(df[target])
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx_tr, idx_val in folds.split(all_groups, all_groups):
        idx_tr_new = df.iloc[idx_tr]
        idx_val_new = df.iloc[idx_val]
        print(len(idx_tr_new),  len(idx_val_new))
        yield idx_tr_new, idx_val_new
    
def main(config):
    opts = config()
    
    comm = chainermn.create_communicator(opts.communicator)
    device = comm.intra_rank

    backborn_cfg = opts.backborn_cfg
    
    df = pd.read_csv(opts.path_data+opts.train_df).sample(frac=1)
    
    ################### pseudo labeling #########################
    if opts.pseudo_labeling_path is not None:
        test_df = pd.read_csv(opts.path_data+opts.test_df)
        labels = np.load(opts.pseudo_labeling_path, allow_pickle=False)
        labels = np.concatenate((labels, labels))
        count = 0
        valid_array = []
        valid_sirna = []
        for i, label in enumerate(labels):
            if label.max() > 0.0013:
                count = count + 1
                valid_array.append(i)
                valid_sirna.append(label.argmax())
        print(count)
        pseudo_df = test_df.iloc[valid_array, :]
        pseudo_df["sirna"] = valid_sirna
        pseudo_df = pseudo_df
        df = pd.concat([df, pseudo_df]).sample(frac=1)
    ################### pseudo labeling #########################
    
    
    
    for i, (train_df, valid_df) in enumerate(stratified_groups_kfold(df, target = opts.fold_target, n_splits=opts.fold)):
        if comm.rank == 0:
            train_df.to_csv(opts.path_data+'train'+'_fold' + str(i)+'.csv', columns=['id_code', 'experiment', 'plate', 'well' ,'sirna', 'filename', 'cell' ,'site'])
            valid_df.to_csv(opts.path_data+'valid'+'_fold' + str(i)+'.csv', columns=['id_code', 'experiment', 'plate', 'well' ,'sirna', 'filename', 'cell' ,'site'])
            print("Save a csvfile of fold_" + str(i))
        dataset = opts.dataset
        train_dataset = dataset(train_df, opts.path_data)
        val_dataset = dataset(valid_df, opts.path_data)
        
        backborn = chcv2_get_model(backborn_cfg['name'], pretrained=backborn_cfg['pretrain'], in_size=opts.input_shape)[backborn_cfg['layer']]
        
        model = opts.model(backborn=backborn).copy(mode='init')
        if device >= 0:
            chainer.cuda.get_device(device).use()
            model.to_gpu()
            
        mean = opts.mean
        
        train_data = TransformDataset(train_dataset, opts.train_transform)
        val_data = TransformDataset(val_dataset, opts.valid_trainsform)
        
        if comm.rank == 0:
            train_indices = train_data
            val_indices = val_data
        else:
            train_indices = None
            val_indices = None
        
        train_data = chainermn.scatter_dataset(train_indices, comm, shuffle=True)
        val_data = chainermn.scatter_dataset(val_indices, comm, shuffle=False)
        train_iter = chainer.iterators.MultiprocessIterator(train_data, opts.batchsize, shuffle=True, n_processes=opts.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(val_data, opts.batchsize, repeat=False, shuffle=False, n_processes=opts.loaderjob)
        print('finished loading dataset') 
        
        if device >= 0:
            chainer.cuda.get_device(device).use()
            model.to_gpu()
        if opts.optimizer == "CorrectedMomentumSGD":
            optimizer = chainermn.create_multi_node_optimizer(CorrectedMomentumSGD(lr = opts.lr), comm)
        elif opts.optimizer == "NesterovAG":
            optimizer = chainermn.create_multi_node_optimizer(NesterovAG(lr = opts.lr), comm)
        else:
            optimizer = chainermn.create_multi_node_optimizer(Adam(alpha=opts.alpha, weight_decay_rate=opts.weight_decay, adabound=True, final_lr=0.5), comm)
   
        optimizer.setup(model)
        if opts.optimizer == "CorrectedMomentumSGD":
            for param in model.params():
                if param.name not in ('beta', 'gamma'):
                    param.update_rule.add_hook(WeightDecay(opts.weight_decay))
                    
        
        if opts.fc_lossfun == 'softmax_cross_entropy':
            fc_lossfun = F.softmax_cross_entropy
        elif opts.fc_lossfun == 'focal_loss':
            if opts.ls:
                focal_loss = FocalLoss(label_smoothing=True)
            else:
                focal_loss = FocalLoss()
            fc_lossfun = focal_loss.loss
        elif opts.fc_lossfun == 'auto_focal_loss':
            if opts.ls:
                focal_loss = AutoFocalLoss(label_smoothing=True)
            else:
                focal_loss = AutoFocalLoss()
            fc_lossfun = focal_loss.loss
        elif opts.fc_lossfun == 'auto_focal_loss_bce':
            if opts.ls:
                focal_loss = AutoFocalLossBCE(label_smoothing=True)
            else:
                focal_loss = AutoFocalLoss()
            fc_lossfun = focal_loss.loss
        if opts.metric_lossfun == 'arcface':
            arcface = ArcFace()
            metric_lossfun = arcface.loss
        elif opts.metric_lossfun == 'adacos':
            adacos = AdaCos()
            metric_lossfun = adacos.loss
            
        updater = opts.updater(train_iter, optimizer, model, device=device, max_epoch=opts.max_epoch, fix_sche = opts.fix_sche, metric_lossfun=metric_lossfun, fc_lossfun = fc_lossfun, metric_w = opts.metric_w, fc_w = opts.fc_w)
        evaluator = chainermn.create_multi_node_evaluator(opts.evaluator(val_iter, model, device=device, max_epoch=opts.max_epoch, fix_sche = opts.fix_sche, metric_lossfun=metric_lossfun, fc_lossfun = fc_lossfun, metric_w = opts.metric_w, fc_w = opts.fc_w), comm)
    
        trainer = training.Trainer(updater, (opts.max_epoch, 'epoch'), out=opts.out+ '_fold' + str(i))
        
        if opts.optimizer == "CorrectedMomentumSGD":
            trainer.extend(extensions.ExponentialShift('lr', opts.shift_lr), trigger=ManualScheduleTrigger(opts.lr_points, 'epoch'))
        elif opts.optimizer == "NesterovAG":
            trainer.extend(extensions.ExponentialShift('lr', opts.shift_lr), trigger=ManualScheduleTrigger(opts.lr_points, 'epoch'))            
        else:
            trainer.extend(extensions.ExponentialShift('alpha', opts.shift_lr), trigger=ManualScheduleTrigger(opts.lr_points, 'epoch'))
        
        trainer.extend(evaluator, trigger=(int(opts.max_epoch/10), 'epoch'))
#         trainer.extend(evaluator, trigger=(int(1), 'epoch'))
        log_interval = 0.1, 'epoch'
        print_interval = 0.1, 'epoch'

        if comm.rank == 0:
            trainer.extend(chainer.training.extensions.observe_lr(), trigger=log_interval)
            trainer.extend(extensions.snapshot_object(model, 'snapshot_model'+'_{.updater.epoch}.npz'), trigger=(opts.max_epoch/10, 'epoch'))
            trainer.extend(extensions.snapshot_object(model, 'snapshot_model_f1max.npz'), 
                           trigger=chainer.training.triggers.MaxValueTrigger( 'validation/main/accuracy', trigger=(opts.max_epoch/10, 'epoch')))
            trainer.extend(extensions.LogReport(trigger=log_interval))
            trainer.extend(extensions.PrintReport(
                ['iteration', 'epoch', 'elapsed_time', 'lr',
                 'main/loss', 'main/face_loss', 'main/ce_loss', 'main/accuracy', 
                 'validation/main/loss','validation/main/face_loss', 'validation/main/ce_loss', 'validation/main/accuracy']
            ), trigger=print_interval)
            trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)