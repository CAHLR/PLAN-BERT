from keras import backend as K
import tensorflow as tf
import keras
import numpy as np

'''
def batch_crossentropy(y_true, y_pred, use_pred):
    use_pred = use_pred[:, :, 0]
    cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred)
    extracted_cross_entropy = K.exp(K.sum(use_pred * cross_entropy, axis=-1) / (K.sum(use_pred, axis=-1) + 1e-6))
    return K.mean(extracted_cross_entropy)
'''
def batch_crossentropy(y_true, y_pred, use_pred):
    use_pred = use_pred[:, :, 0]
    cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(use_pred * cross_entropy, axis=-1) / (K.sum(use_pred, axis=-1) + 1e-6)
    return K.mean(cross_entropy)


def confidence_penalty(y_pred, use_pred):
    use_pred = use_pred[:, :, 0]
    return K.mean(K.sum(y_pred * K.log(y_pred), axis=-1) * use_pred)


def recall_at_10(y_true, y_pred, use_pred):
    use_pred = use_pred[:, :, 0]
    # Tenth largest elements (output probs) over last axis 
    _, indices = tf.nn.top_k(y_pred, 10) # batch_size * num_semesters * 10
    # Multihot mask of top ten predictions (output classes) from y_pred
    y_pred_top = K.sum(K.one_hot(indices, K.int_shape(y_pred)[-1]), axis=-2) # batch_size * num_semesters * 10 * num_courses
    hit_mat = K.round(K.clip(y_true * y_pred_top, 0, 1))
    true_mat = K.round(K.clip(y_true, 0, 1))
    
    hit_sum = K.sum(K.sum(hit_mat, axis=-1) * use_pred)
    true_sum = K.sum(K.sum(true_mat, axis=-1) * use_pred)
    # recall_at_10 = 1 if there are no positive examples
    return hit_sum / (true_sum + K.epsilon())