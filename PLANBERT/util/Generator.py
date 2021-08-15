import os, pickle, tqdm, keras, time
import numpy as np
import pandas as pd
from copy import deepcopy
from . import Datahelper as dh


class TimeMultihotGenerator(keras.utils.Sequence):
    def __init__(self, user_dict, user_ids, basic_config, generator_config):
        super(TimeMultihotGenerator, self).__init__()

        self.name = generator_config['name']
        self.user_dict, self.user_ids = user_dict, user_ids
        self.basic_config = basic_config
        self.generator_config = generator_config

        self.num_items = self.basic_config['num_items']
        self.num_times = self.basic_config['num_times']
        self.feats = basic_config['feats']

        self.pool_sampling = self.generator_config['pool_sampling']
        self.pool_size = 0 if not self.pool_sampling else self.generator_config['pool_size']
        
        self.sample_func = self.generator_config['sample_func']
        self.history_func = self.generator_config['history_func']
        self.batch_size = self.generator_config['batch_size']
        self.shuffle = self.generator_config['shuffle']
        self.fixed_seed = self.generator_config['fixed_seed']
        
        self.num_epoch = 0
        self.on_epoch_end()
        
    def __len__(self):
        return int(len(self.user_ids) / self.batch_size)
    
    def __getitem__(self, batch_id, batch_size=None):
        if batch_size == None:
            user_ids = self.user_ids[(batch_id * self.batch_size):((batch_id + 1) * self.batch_size)]
        elif batch_size == 'MAX':
            user_ids = self.user_ids
        else:
            user_ids = self.user_ids[(batch_id * batch_size):((batch_id + 1) * batch_size)]

        input_list = []
        for user_id in user_ids:
            input = self.data_generation(user_id)
            if len(input_list) == 0:
                for iter in range(len(input)):
                    input_list.append([input[iter][np.newaxis]])
            else:
                for iter in range(len(input)):
                    input_list[iter].append(input[iter][np.newaxis])
                    
        input_list = [np.concatenate(iter, axis=0) for iter in input_list]
        return input_list, []
    
    
    def on_epoch_end(self):
        self.num_epoch += 1
        if self.shuffle == True:
            # Shuffle the order of inputs.
            np.random.seed(self.num_epoch)
            np.random.shuffle(self.user_ids)
            
            
    def data_generation(self, id, start_time=None):
        # If use fixed seed: use user's id as random seed.
        # Else: use user's id + epoch number as random seed.
        seed = int(id) if self.fixed_seed else (int(id) + self.num_epoch)
        np.random.seed(seed)
        
        tabular = np.array(self.user_dict[id])

        if start_time is None:
            start_time = 0

        tabular = tabular[
            (tabular[:, 0] >= start_time) & \
            (tabular[:, 0] < (start_time + self.num_times))]
        tabular[:, 0] -= start_time
        tab_row = tabular.shape[0]

        sample_num = int(eval(self.sample_func)(tab_row))
        history_num = int(eval(self.history_func)(self.num_times))
        history_ids = list(np.where(tabular[:, 0] < history_num)[0])

        if self.pool_sampling:
            temp, remain_ids = dh.list_sampling(np.arange(tab_row).astype(int), self.pool_size)
            sample_ids = dh.list_sampling(temp, sample_num)[0]
        else:
            sample_ids, remain_ids = dh.list_sampling(np.arange(tab_row).astype(int), sample_num)
            
        known_ids = np.unique(list(history_ids) + list(sample_ids)).astype(int)
        unknown_ids = list(filter(lambda x:x not in history_ids, remain_ids))
        known_rows, unknown_rows = tabular[known_ids], tabular[unknown_ids]
        
        # ItemID
        x0 = dh.list2mat(known_rows[:, [0, 1]], [self.num_times, self.num_items])
        # Predict Token
        x1 = np.concatenate(
            [np.zeros([history_num]), np.ones([self.num_times - history_num])])[:,np.newaxis]
        # Relative Semester
        x2 = np.eye(self.num_times)
        # Features
        feat_list = [dh.list2mat(tabular[history_ids][:, [0, iter + 2]], [self.num_times, self.feats[iter]]) for iter in range(len(self.feats))]
        # Target
        y = dh.list2mat(unknown_rows[:, [0, 1]], [self.num_times, self.num_items])
        return [x0, x1, x2, *feat_list, y]