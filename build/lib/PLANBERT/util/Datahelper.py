import numpy as np
import tqdm, random, copy


def onehot_encoding(List, shape):
    List_2 = np.zeros(shape)
    List_2[np.arange(len(List)).astype(int), np.array(List).astype(int)] = 1
    return List_2

def multihot_encoding(List, shape):
    List_2 = np.zeros(shape)
    for x, iter in enumerate(List):
        List_2[x, iter] = 1
    return List_2

def list_partition(List, proportion, seed=0):
    # return the partitioned list.
    assert (proportion >= 0) and (proportion <= 1)
    List_2 = copy.deepcopy(List)
    
    np.random.seed(seed)
    np.random.shuffle(List_2)
    
    List_1 = List[:int(len(List) * proportion)]
    List_2 = List[int(len(List) * proportion):]
    return List_1, List_2


def list_sampling(List, sample_num, seed=0):
    if len(List) == 0:
        return np.array(List), np.array(List)
    list_len = len(List)
    list_index_list = np.arange(list_len)
    
    np.random.seed(seed)
    np.random.shuffle(list_index_list)
    
    return List[list_index_list[:sample_num]], List[list_index_list[sample_num:]]


def mat_partition(mat, time_slice):
    # Split a matrix as history and future.
    # input: (Timeslice, CourseNum)
    X = np.zeros(mat.shape)
    Y = np.zeros(mat.shape)
    
    X[:time_slice, :] = mat[:time_slice, :]
    Y[time_slice:, :] = mat[time_slice:, :]
    
    return X, Y


def mat_sampling(mat, sample_num, window, seed=0):
    # Sample from a matrix.
    # input: (Timeslice, CourseNum)
    X = np.zeros(mat.shape)
    
    index = np.array(np.where(mat))
    index = index[:, ((index[0, :] >= window[0]) * (index[0, :] < window[1]))]
    index = index.T
    
    np.random.seed(seed)
    np.random.shuffle(index)
    
    index = index.T[:, :sample_num]
    X[index[0, :], index[1, :]] = 1
    Y = mat - X
    
    return X, Y


def list_padding(List, target_len, token):
    List_2 = np.array([token] * target_len)
    padding_index = np.ones([target_len, ])
    
    List_2[:len(List)] = np.array(List)
    padding_index[:len(List)] = 0
    return List_2, padding_index


def mat_padding(mat_1, shape, token):
    mat_2 = np.zeros(shape) + token
    mat_2[:mat_1.shape[0], :mat_2.shape[1]] = mat_1
    return mat_2


def set_top_n(arr, n):
    # return the multi-hot vector of the position of top n.
    # input: 1-D vector
    arr2 = np.zeros(arr.shape)
    arr2[np.argsort(arr)[-n:]] = 1
    return arr2


def list2mat(list1, shape):
    mat = np.zeros(shape)
    if len(list1.shape) == 2:
        if list1.shape[0] > 0:
            mat[list1[:, 0].astype(int), list1[:, 1].astype(int)] = 1
    return mat