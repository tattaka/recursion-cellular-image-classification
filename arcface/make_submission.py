from __future__ import division
import argparse
import multiprocessing
import numpy as np
import re
import sys
from sklearn.model_selection import StratifiedKFold
from tqdm import trange

import copy

import chainer
from chainer import functions as F
chainer.config.autotune = True
chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)

from chainercv.chainer_experimental.datasets.sliceable import TransformDataset

from chainercv2.model_provider import get_model as chcv2_get_model

from functools import partial

import pandas as pd

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

import argparse 
from importlib import import_module

def main(config):
    opts = config()
    device = 0

    backborn_cfg = opts.backborn_cfg
    
    test_df = pd.read_csv(opts.path_data+opts.test_df)
   
        
    test_dataset = opts.dataset(test_df, opts.path_data, mode='test')

    backborn = chcv2_get_model(backborn_cfg['name'], pretrained=backborn_cfg['pretrain'], in_size=opts.input_shape)[backborn_cfg['layer']]

    model = opts.model(backborn=backborn)

    mean = opts.mean
    
    test_data = TransformDataset(test_dataset, opts.valid_trainsform)
    test_data_flip1 = TransformDataset(test_dataset, opts.valid_trainsform_flip1)
    test_data_flip2 = TransformDataset(test_dataset, opts.valid_trainsform_flip2)
    test_data_flip3 = TransformDataset(test_dataset, opts.valid_trainsform_flip3)

    print('finished loading dataset') 

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()
        
    result = np.zeros((int(test_data.__len__()/2), opts.num_class))
    test_len = int(test_data.__len__())
    for fold in trange(opts.fold, desc='fold loop'):
        chainer.serializers.load_npz(opts.out+'_fold'+str(fold)+'/snapshot_model_f1max.npz', model)
        for i in trange(test_len, desc='id loop'):
            for _ in range(4):
                img, img_id = test_data.get_example(i)
                img = img[None, :, :, :]
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    res = chainer.cuda.to_cpu(F.softmax(model(img)[1]).data)
                result[i%int(test_len/2)] = result[i%int(test_len/2)] + res
                ############################### TTA ###############################
                img, img_id = test_data_flip1.get_example(i)
                img = img[None, :, :, :]
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    res = chainer.cuda.to_cpu(F.softmax(model(img)[1]).data)
                result[i%int(test_len/2)] = result[i%int(test_len/2)] + res
                img, img_id = test_data_flip2.get_example(i)
                img = img[None, :, :, :]
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    res = chainer.cuda.to_cpu(F.softmax(model(img)[1]).data)
                result[i%int(test_len/2)] = result[i%int(test_len/2)] + res
                img, img_id= test_data_flip3.get_example(i)
                img = img[None, :, :, :]
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    res = chainer.cuda.to_cpu(F.softmax(model(img)[1]).data)
                result[i%int(test_len/2)] = result[i%int(test_len/2)] + res
                ##################################################################
        
    result = result / opts.fold / 32
    probs = result
    np.save(opts.out+"_probs.npy", probs)
    result_t = np.argmax(result, axis = 1)
    print(result_t.shape)
    submission = pd.read_csv(opts.path_data + '/test.csv')
    submission['sirna'] = result_t.astype(int)
    submission.to_csv('submission_'+opts.out+'.csv', index=False, columns=['id_code','sirna'])
    
    ############################### Use leak #########################################
    train_csv = pd.read_csv("../input/train.csv")
    test_csv = pd.read_csv("../input/test.csv")
    sub = pd.read_csv('submission_'+opts.out+'.csv')
    
    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum()
    
    all_test_exp = test_csv.experiment.unique()
    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna']
        pp_mult = np.zeros((len(preds),1108))
        pp_mult[range(len(preds)),preds] = 1

        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                   np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
    exp_to_group = group_plate_probs.argmax(1)
    
    def select_plate_group(pp_mult, idx):
        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
               np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        pp_mult[mask] = 0
        return pp_mult
    
    for idx in range(len(all_test_exp)):
        #print('Experiment', idx)
        indices = (test_csv.experiment == all_test_exp[idx])

        preds = result[indices,:].copy()

        preds = select_plate_group(preds, idx)
        result[indices, :] = preds
    probs_leak = result
    np.save(opts.out+"_probs_leak.npy", probs_leak)
    result = np.argmax(result, axis = 1)
    print(result.shape)
    submission = pd.read_csv('submission_'+opts.out+'.csv')
    submission['sirna'] = result.astype(int)
    submission.to_csv('submission_leak_'+opts.out+'.csv', index=False, columns=['id_code','sirna'])
    ###################################################################################
    ############################### Use Hungarian Algorithm##################################
    import scipy
    import scipy.special
    import scipy.optimize
    
    def assign_plate(plate):
        probabilities = np.array(plate)
        cost = probabilities * -1
        rows, cols = scipy.optimize.linear_sum_assignment(cost)
        chosen_elements = set(zip(rows.tolist(), cols.tolist()))

        for sample in range(cost.shape[0]):
            for sirna in range(cost.shape[1]):
                if (sample, sirna) not in chosen_elements:
                    probabilities[sample, sirna] = 0

        return probabilities
    current_plate = None
    plate_probabilities = []
    probs_hungarian = []
    for i, name in tqdm(enumerate(submission['id_code'])):
        experiment, plate, _ = name.split('_')
        if plate != current_plate:
            if current_plate is not None:
                probs_hungarian.extend([x for x in assign_plate(plate_probabilities)])
            plate_probabilities = []
            current_plate = plate
        plate_probabilities.append(scipy.special.softmax(probs_leak[i]))
    probs_hungarian.extend([x for x in assign_plate(plate_probabilities)])
    
    np.save(opts.out+"_probs_leak_hungarian.npy", probs_hungarian)
    result = np.argmax(result, axis = 1)
    print(result.shape)
    submission = pd.read_csv('submission_'+opts.out+'.csv')
    submission['sirna'] = result.astype(int)
    submission.to_csv('submission_leak_hungarian'+opts.out+'.csv', index=False, columns=['id_code','sirna'])
    
    
    return probs, probs_leak, probs_hungarian

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)