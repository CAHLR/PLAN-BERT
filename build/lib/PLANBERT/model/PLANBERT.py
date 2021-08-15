from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Multiply, Lambda, Dense, Masking
from keras.optimizers import Adam
import numpy as np

from .transformer_util.extras import TiedOutputEmbedding
from .transformer_util.transformer import TransformerBlock
from .transformer_util.multihot_utils import ReusableEmbed_Multihot
from .transformer_util.attention import MultiHeadSelfAttention
from .LossFunction import batch_crossentropy, confidence_penalty, recall_at_10


def Transformer(config):
    mask_future = config['mask_future'] # False
    vanilla_wiring = True # True
    use_parameter_sharing = False # False

    num_times = config['num_times']
    width = config['width'] if 'width' in config else config['num_times']
    num_input_list = config['base_feats']
    num_input_list += config['feats']

    num_layers = config['num_layers']
    embedding_dim = config['embedding_dim']
    num_heads = config['num_heads']

    transformer_dropout = config['transformer_dropout']
    embedding_dropout = config['embedding_dropout']
    l2_reg_penalty_weight = config['l2_reg_penalty_weight']
    confidence_penalty_weight = config['confidence_penalty_weight']
    lrate = config['lrate']

    # [WhetherTheFeatureIsUsed, DimOfFeature, Name, InputLayer]
    # [WhetherTheFeatureIsUsed, DimOfFeature, Name, InputLayer]
    num_input = num_input_list[0]
    num_input_list[0].append(Input(shape=(width, num_input[1]), dtype='float', name=num_input[2]))
    for iter, num_input in enumerate(num_input_list[1:]):
        num_input_list[iter+1].append(Input(shape=(width,num_input[1]), dtype='float', name=num_input[2]))

    l2_reg = (regularizers.l2(l2_reg_penalty_weight) if l2_reg_penalty_weight else None)

    embedding, embedding_matrix = ReusableEmbed_Multihot(
            num_input_list[0][1], embedding_dim, input_length=width,name=num_input_list[0][2]+'Embedding',embeddings_regularizer=l2_reg)(num_input_list[0][-1])
    embedding_list = [embedding]
    for num_input in num_input_list[1:]:
        if num_input[0] == True: # if the feature is used
            embedding_list.append(
                ReusableEmbed_Multihot(
                    num_input[1], embedding_dim, input_length=width, name=num_input[2]+'Embedding'
                )(num_input[3])[0]
            )
    target = Input(shape=(width, num_input_list[0][1]), dtype='float', name='Target')
    use_pred = num_input_list[1][-1]
    next_step_input = Add(name='Embedding_Add')(embedding_list)

    if use_parameter_sharing:
        attention_layer = MultiHeadSelfAttention(
            num_heads, use_masking=mask_future, dropout=transformer_dropout, compression_window_size=None, name='self_attention')
    else:
        attention_layer = None
    # Transformer Architecture.
    for i in range(num_layers):
        next_step_input = TransformerBlock(
            name='transformer_'+str(i),
            num_heads=num_heads,
            residual_dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
            use_masking=mask_future,
            vanilla_wiring=vanilla_wiring,
            attention_layer=attention_layer)(next_step_input)

    predict = Softmax(name='word_predictions_0')(TiedOutputEmbedding(
        projection_regularizer=l2_reg,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits_1')([next_step_input, embedding_matrix]))

    loss_layer_1 = Lambda(lambda x: batch_crossentropy(*x), name="loss_layer_1")([target, predict, use_pred])
    loss_layer_2 = Lambda(lambda x: confidence_penalty_weight * confidence_penalty(*x), name="loss_layer_2")([predict, use_pred])
    metric_layer_1 = Lambda(lambda x: recall_at_10(*x), name="metric_layer_1")([target, predict, use_pred])

    outputs = [predict, loss_layer_1, loss_layer_2, metric_layer_1]

    input_list = [each[-1] for each in num_input_list]
    input_list.append(target)
    model = Model(inputs=input_list, outputs=outputs)

    loss_1 = model.get_layer("loss_layer_1").output
    model.add_loss(loss_1)
    loss_2 = model.get_layer("loss_layer_2").output
    model.add_loss(loss_2)
    metric_1 = model.get_layer("metric_layer_1").output
    model.add_metric(metric_1, name='recall_at_10')
    
    # Deploying Optimizer.
    model.compile(
        optimizer=Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0),
        loss=[None, None, None, None]
    )

    return model
