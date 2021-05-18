import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import trange
import sys, os, pickle, importlib, keras, random, tqdm, copy, json, time, argparse
from keras import backend as K

from . import Engine
from .util import Datahelper as dh
from .util import Generator as Generator
from .model import PLANBERT as Transformer


def dataframe2dict(csv):
    user_dict = {}
    for iter in trange(csv.shape[0]):
        user = int(csv.iloc[iter][0])
        if user not in user_dict:
            user_dict[user] = []
        user_dict[user].append(list(csv.iloc[iter][1:]))
    for key in user_dict:
        user_dict[key] = np.array(user_dict[key])
    return user_dict


class model:
    def __init__(self,
                 num_times, num_items, feat_dims, cuda_num, master_csv=None,
                 num_layers=2, num_hidden_dims=2**9, num_heads=8,
                 transformer_dropout=0.2, embedding_dropout=0.2,
                 l2_reg_penalty_weight=0, confidence_penalty_weight=0.1,
                 lrate=1e-4, seed=0,
                 checkpoint=None
                ):

        if master_csv:
            self.__num_times = master_csv.iloc[:, 1].max() + 1
            self.__num_items = master_csv.iloc[:, 2].max() + 1
            self.__feat_dims = [master_csv.iloc[:, iter].max() + 1 for iter in range(3, master_csv.shape[1])]
        else:
            self.__num_times = num_times
            self.__num_items = num_items
            self.__feat_dims = feat_dims

        self.__num_layers = num_layers
        self.__num_hidden_dims = num_hidden_dims
        self.__num_heads = num_heads

        self.__transformer_dropout = transformer_dropout
        self.__embedding_dropout = embedding_dropout
        self.__l2_reg_penalty_weight = l2_reg_penalty_weight
        self.__confidence_penalty_weight = confidence_penalty_weight

        self.__lrate = lrate
        self.__seed = seed
        self.__cuda_num = cuda_num
        self.__checkpoint = checkpoint

        self.__model_config = {'num_times': self.__num_times, 'num_items': self.__num_items, 'feats': feat_dims}

        # Training Environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.__cuda_num)
        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth=True
        session = tf.compat.v1.Session(config=session_config)
        #Engine.set_random_seed(self.__seed)

        # Build Model
        temp_config = {
            'name': 'PLANBERT',
            'mask_future': False,
            'num_times': self.__num_times, 'num_items': 0,
            'base_feats': [[True, self.__num_items, 'ItemID'], [True, 1, 'PredictToken'], [True, self.__num_times, 'Time']],
            # [whether the feature is used, the dimension of the feature, the name of feature]
            'feats': [[True, self.__feat_dims[iter], 'Feat{}'.format(iter)] for iter in range(len(self.__feat_dims))],
            'embedding_dim': self.__num_hidden_dims,
            'num_layers': self.__num_layers,
            'num_heads': self.__num_heads,
            'transformer_dropout': self.__transformer_dropout, 'embedding_dropout': self.__embedding_dropout,
            'l2_reg_penalty_weight': self.__l2_reg_penalty_weight, 'confidence_penalty_weight': self.__confidence_penalty_weight, 'lrate': self.__lrate
        }

        self.model = Transformer.Transformer(temp_config)
        if self.__checkpoint:
            self.model.load_weights(self.__checkpoint)

    def fit(
        self, train_data, valid_data, pt_percentage=0.8,
        nonimprove_limit=10, batch_size=32, shuffle=True, fixed_seed=False, epoch_limit=500
    ):
        if type(train_data) == pd.DataFrame:
            train_dict = dataframe2dict(train_data)
        else:
            assert type(train_data) == dict
            train_dict = train_data

        if type(valid_data) == pd.DataFrame:
            valid_dict = dataframe2dict(valid_data)
        else:
            assert type(valid_data) == dict
            valid_dict = valid_data

        train_generator_config = {
            'name': None,
            'pool_sampling': False,
            'sample_func': '(lambda x: {} * x)'.format(pt_percentage),
            'history_func': '(lambda x: 0.0 * x)',
            'batch_size': batch_size,
            'shuffle': shuffle,
            'fixed_seed': fixed_seed}

        train_generator = Generator.TimeMultihotGenerator(
            train_dict, list(train_dict.keys()), self.__model_config, train_generator_config)
        valid_generator = Generator.TimeMultihotGenerator(
            valid_dict, list(valid_dict.keys()), self.__model_config, train_generator_config)

        Engine.fit(
            model=self.model,
            train_generator=train_generator,
            valid_generator=valid_generator,
            epoch_limit=epoch_limit, loss_nonimprove_limit=nonimprove_limit, batch_size=batch_size,
            use_cosine_lr=True, model_save_path=None)

        # Fine-tune
        train_generator_config = {
            'name': None,
            'pool_sampling': False,
            'sample_func': '(lambda x: np.random.randint(x))',
            'history_func': '(lambda x: np.random.randint(x))',
            'batch_size': batch_size,
            'shuffle': shuffle, 'fixed_seed': fixed_seed}

        train_generator = Generator.TimeMultihotGenerator(
            train_dict, list(train_dict.keys()), self.__model_config, train_generator_config)
        valid_generator = Generator.TimeMultihotGenerator(
            valid_dict, list(valid_dict.keys()), self.__model_config, train_generator_config)

        Engine.fit(
            model=self.model,
            train_generator=train_generator,
            valid_generator=valid_generator,
            epoch_limit=epoch_limit,
            loss_nonimprove_limit=nonimprove_limit,
            batch_size=batch_size,
            use_cosine_lr=True, model_save_path=None)


    def save(self, path):
        self.model.save_weights(path)


    def predict(self, test_data, mode, history_dict):
        # history_dict: { user_id: num_history }

        if type(test_data) == pd.DataFrame:
            test_dict = dataframe2dict(test_data)
        else:
            assert type(test_data) == dict
            test_dict = test_data

        assert(mode in ['time', 'wishlist'])
        test_generator_config = {'name':'None', 'pool_sampling': False, 'batch_size': 1, 'shuffle': False, 'fixed_seed': True}

        result_dict = {}
        for iter in history_dict:
            num_history = history_dict[iter]
            test_generator_config['sample_func'] = '(lambda x:x)'
            test_generator_config['history_func'] = '(lambda x:{})'.format(num_history)
            test_generator = Generator.TimeMultihotGenerator(
                test_dict, [iter], self.__model_config, test_generator_config)
            result_dict[iter] = Engine.predict(
                self.model, test_generator, pred_window=[num_history, self.__num_times])
        return result_dict


    def test(self, test_data, h_list, r_list, pool_size=None, batch_size=16):
        if type(test_data) == pd.DataFrame:
            test_dict = dataframe2dict(test_data)
        else:
            assert type(test_data) == dict
            test_dict = test_data

        if pool_size==None:
            pool_size=np.max(r_list)
        test_generator_config = {
            'pool_sampling': True, 'pool_size': pool_size, 'batch_size': batch_size,
            'shuffle': False, 'fixed_seed': True
        }
        results_mat, wishlist_mat = {}, {}
        for h in h_list:
            results_mat[h], wishlist_mat[h] = {}, {}
            for r in r_list:
                test_generator_config['sample_func'] = '(lambda x:{})'.format(r)
                test_generator_config['history_func'] = '(lambda x:{})'.format(h)
                test_generator_config['name'] = 'H={0}_R={1}'.format(h, r)
                test_generator = Generator.TimeMultihotGenerator(
                    test_dict, list(test_dict.keys()), self.__model_config, test_generator_config)

                #print(test_generator.name)
                recall, recall_per_sem = Engine.test(self.model, test_generator, pred_window=[h, self.__num_times])
                results_mat[h][r] = [recall, recall_per_sem]
                recall, recall_per_sem = Engine.test_wishlist(self.model, test_generator, pred_window=[h, self.__num_times])
                wishlist_mat[h][r] = [recall, recall_per_sem]
        return results_mat, wishlist_mat
