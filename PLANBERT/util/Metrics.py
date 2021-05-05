import numpy as np
import copy
from tqdm import tqdm


def recall(target, predict, at_n):
    num_students = target.shape[0]
    num_semesters = target.shape[1]
    num_courses = target.shape[2]
    
    recall_list = []
    recall_per_sem_list = []
    
    for sample in range(num_students):
        if target[sample].sum() > 0:
            hit_mat = np.zeros([num_semesters, num_courses])
            for semester in range(num_semesters):
                n_temp = target[sample, semester].sum()
                if len(np.shape(at_n)) == 1: # recall@n on average
                    index = np.argsort(-predict[sample, semester])[:int(at_n[semester])]
                elif at_n == 0: # recall@n
                    index = np.argsort(-predict[sample, semester])[:int(n_temp)]
                elif at_n > 0: # recall@10
                    index = np.argsort(-predict[sample, semester])[:int(at_n)]
                hit_mat[semester, index] = 1
            hit_mat *= target[sample]
            recall_list.append(hit_mat)
        else:
            continue
    return np.array(recall_list).sum() / target.sum(), np.array(recall_list).sum(0).sum(-1) / target.sum(0).sum(-1)


def MRR(target, predict, at_n=None):
    rank = predict.argsort(axis=-1)[:,:,::-1]
    if at_n is not None:
        rank = rank[:,:,:at_n]
        
    mmr_mat = np.zeros_like(target)
    for r in range(rank.shape[-1]):
        for sem in range(target.shape[1]):
            mmr_mat[np.arange(target.shape[0]), sem, rank[:, sem, r]] += 1/(r+1)
    mmr_mat *= target
    
    mmr_per_time_slot = mmr_mat.sum(-1).mean(0)
    mean_mrr = mmr_per_time_slot.mean()
    
    return mmr_mat.sum(-1)


def DCG(target, predict, at_n=None):
    rank = predict.argsort(axis=-1)[:,:,::-1]
    if at_n is not None:
        rank = rank[:,:,:at_n]
    DCG_mat = np.zeros_like(target)
    for sem in range(target.shape[1]):
        r = 0
        DCG_mat[np.arange(target.shape[0]), sem, rank[:, sem, r]] = 1
    for r in range(1, rank.shape[-1]):
        for sem in range(target.shape[1]):
            DCG_mat[np.arange(target.shape[0]), sem, rank[:, sem, r]] = 1 / np.log2(r+2)
    DCG_mat *= target
    return DCG_mat.sum(-1)


def NDCG(target, predict, at_n=None):
    DCG_mat = DCG(target, predict, at_n=at_n).sum(-1)
    IDCG_mat = DCG(target, target, at_n=at_n).sum(-1)
    valid = IDCG_mat !=0
    NDCG_mat = DCG_mat[valid] / IDCG_mat[valid]
    return NDCG_mat