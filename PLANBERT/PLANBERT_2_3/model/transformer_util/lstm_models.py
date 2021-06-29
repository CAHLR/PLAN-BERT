import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import Adam
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense, Softmax
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.multihot_utils import ReusableEmbed_Multihot

import keras.callbacks as callbacks
import keras.backend as K

def lstm_model(self, lrate=0.01, layers=2, embed_dim=128, seq_len=256, embedding_vocab_size=4000, confidence_penalty_weight=0, use_tied_embedding=False, lstm_dropout=0.2, model_load_path=None):
    """
    Returns a LSTM model
    """
    self.model_params = {'layers': layers,'embed_dim': embed_dim,'e_vocab_size': self.embedding_vocab_size,'seq_len': seq_len,'lrate': lrate}

    # Input Layer
    if self.multihot_input:
        main_input = Input(shape=(self.model_params['seq_len'], self.model_params['e_vocab_size']), dtype='float', name='onehot_ids')
    else:
        main_input = Input(shape=(self.model_params['seq_len'],), dtype='int32', name='node_ids')

    # Tied Embedding Layer
    if use_tied_embedding:
        l2_regularizer = keras.regularizers.l2(1e-6)
        if self.multihot_input:
            embedding_layer = ReusableEmbed_Multihot(
                input_dim=self.model_params['e_vocab_size'],
                output_dim=self.model_params['embed_dim'],
                input_length=self.model_params['seq_len'],
                name='multihot_embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                embeddings_regularizer=l2_regularizer)
        else:
            embedding_layer = ReusableEmbedding(
                input_dim=self.model_params['e_vocab_size'],
                output_dim=self.model_params['embed_dim'],
                input_length=self.model_params['seq_len'],
                name='word_embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)

        output_layer = TiedOutputEmbedding(
            projection_regularizer=l2_regularizer,
            projection_dropout=0.6,
            name='word_prediction_logits')
        output_softmax_layer = Softmax(name='word_predictions')
        next_step_input, embedding_matrix = embedding_layer(main_input)
    # Regular Embedding Layer
    else:
        if self.multihot_input:
            embedding_layer = TimeDistributed(Dense(
                    units=self.model_params['embed_dim'],
                    activation=None, use_bias=False,
                    name='multihot_embeddings'))
        else:
            embedding_layer = Embedding(
                    input_dim=self.model_params['e_vocab_size'],
                    output_dim=self.model_params['embed_dim'],
                    input_length=self.model_params['seq_len'],
                    mask_zero=True, name='word_embeddings')
        output_layer = TimeDistributed(Dense(self.model_params['e_vocab_size'], 
            activation='softmax', 
            name='word_predictions'))
        next_step_input = embedding_layer(main_input)

    for i in range(self.model_params['layers']):
        next_step_input = LSTM(self.model_params['embed_dim'], dropout=0.2, return_sequences=True, name='LSTM_layer_{}'.format(i))(next_step_input)

    # Tied Embedding Layer
    if use_tied_embedding:
        word_predictions = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
    else:
        word_predictions = output_layer(next_step_input)

    self.model = Model(inputs=[main_input], outputs=[word_predictions])
        
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    if confidence_penalty_weight > 0:
        confidence_penalty = K.mean(confidence_penalty_weight *
            K.sum(word_predictions * K.log(word_predictions +K.epsilon()), axis=-1))
        self.model.add_loss(confidence_penalty)

    optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
    self.compile_and_load(optimizer, model_load_path)
