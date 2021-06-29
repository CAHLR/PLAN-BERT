"""
Reusable Embedding similar to that from package 'keras_transformer', but tailored to accept multi-hot input instead of token IDs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
#from keras.legacy import interfaces

def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.
    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]

def recall_at_10(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    # Tenth largest elements (output probs) over last axis 
    _, indices = tf.nn.top_k(y_pred, 10) # batch_size * num_semesters * 10
    # Multihot mask of top ten predictions (output classes) from y_pred
    y_pred_top = K.sum(K.one_hot(indices, K.int_shape(y_pred)[-1]), axis=-2) # batch_size * num_semesters * 10 * num_courses
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_top, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # recall_at_10 = 1 if there are no positive examples
    return true_positives / (possible_positives + K.epsilon())
    

class ReusableEmbed_Multihot(Layer):
    '''
    Turns positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    This layer can only be used as the first layer in a model.
    # Example
    ```python
      model = Sequential()
      model.add(ReusableEmbed_Multihot(1000, 64, input_length=10))
      # the model will take as input an integer matrix of size (batch, input_length).
      # the largest integer (i.e. word index) in the input should be
      # no larger than 999 (vocabulary size).
      # now model.output_shape == (None, 10, 64), where None is the batch dimension.
      input_array = np.random.randint(1000, size=(32, 10))
      model.compile('rmsprop', 'mse')
      output_array = model.predict(input_array)
      assert output_array.shape == (32, 10, 64)
    ```
    # Arguments
        input_dim: int > 0. Size of the multihot field,
            i.e. input_shape[-1].
        output_dim: int >= 0. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings` matrix
            (see [initializers](../initializers.md)).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix
            (see [constraints](../constraints.md)).
        mask_zero: Whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using [recurrent layers](recurrent.md)
            which may take variable length input.
            If this is `True` then all subsequent layers
            in the model need to support masking or an exception will be raised.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''

    #@interfaces.legacy_embedding_support
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length, input_dim)
            else:
                kwargs['input_shape'] = (None, input_dim)
        super(ReusableEmbed_Multihot, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        output_mask = K.not_equal(K.sum(inputs, axis=2, keepdims=True), 0)
        return [output_mask, None]

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return [input_shape[:-1] + (self.output_dim,), K.int_shape(self.embeddings)]
        else:
            # input_length can be tuple if input is 3D or higher
            in_lens = to_list(self.input_length, allow_tuple=True)
            if len(in_lens) != len(input_shape) - 2:
                raise ValueError(
                    '"input_length" is %s, but received input has shape %s' %
                    (str(self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:-1])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError(
                            '"input_length" is %s, but received input has shape %s' %
                            (str(self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return [(input_shape[0],) + tuple(in_lens) + (self.output_dim,), K.int_shape(self.embeddings)]

    def call(self, inputs):
        out = K.dot(inputs, self.embeddings)
        return [out, self.embeddings]

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer':
                      regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint':
                      constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(ReusableEmbed_Multihot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
