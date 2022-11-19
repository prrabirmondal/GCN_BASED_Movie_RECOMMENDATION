## it generates the ndcg value for a particular K

import re
from pip import main
from requests import delete
from sklearn.metrics import ndcg_score, dcg_score
import numpy as np

def sort_list(Lst):  
    """sorting the movies according to its ratings values"""
    row, col = Lst.shape
    true_sort = []
    for i in range(0,col):
        max_indx = np.argmax(Lst[1][:])  # returning the index of maximum value of numpy array
        # true_sort.append(list([Lst[0][max_indx], Lst[1][max_indx]]))
        true_sort.append(Lst[0][max_indx])
        Lst = np.delete(Lst, max_indx, axis=1)
    return true_sort

def fndcg_k(true_relevance, sort_PR, k):
    sort_pr = sort_PR[:k]
    _, col = true_relevance.shape  # only the column are taken
    relevance_socre = []

    for i in sort_pr:
        for j in range(0, col):
            if i == true_relevance[0][j]:
                relevance_socre.append(true_relevance[1][j])

    true_score = list(relevance_socre)
    true_score.sort(reverse=True)
#     print("true_score = ", true_score)
#     print("relevance_score = ",relevance_socre)

    ndcg_k = ndcg_score(
        np.array([true_score]), np.array([relevance_socre]))

    return ndcg_k

def my_ndcg(res_true, res_pred, k):

    movie_count = len(res_true)
    index = np.array(range(1,movie_count+1))

    true_relevance = np.asarray([index, res_true])
    pred_relevance = np.asarray([index, res_pred]) 

    sort_PR = sort_list(pred_relevance)  # sorted predicted order

    if k>len(sort_PR):
        return 0
 
    ndcg_k = fndcg_k(true_relevance, sort_PR, k)

    return(ndcg_k)