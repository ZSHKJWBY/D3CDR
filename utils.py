import numpy as np
import random as rd
from collections import defaultdict
import scipy.sparse as sp
import params
import math

def normalize_adj(adj):
    s = 1/adj.sum(1)
    s[np.isnan(s)] = 0.0
    s[np.isinf(s)] = 0.0
    d = np.diag(s)#np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
    a_norm = d.dot(adj)
    return a_norm


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return float(np.sum(r)) / float(all_pos_num)


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def best_result(best, current):
    # print("find the best number:")
    num_ret = len(best)
    ret_best = [0.0]*num_ret
    for numIdx in range(num_ret):
        ret_best[numIdx] = max(float(current[numIdx]), float(best[numIdx]))
    return ret_best


def generate_test(all_user_ratings):
    ratings_test = {}
    for user in all_user_ratings:
        ratings_test[user] = rd.sample(all_user_ratings[user], 1)[0]
    return ratings_test


def generate_train_batch(user_ratings, user_ratings_test, n, batch_size):
    t = []
    for b in range(batch_size):
        u = rd.sample(user_ratings.keys(), 1)[0]
        i = rd.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            if (len(user_ratings[u]) == 1):
                break
            i = rd.sample(user_ratings[u], 1)[0]
        j = rd.randint(0, n - 1)
        while j in user_ratings[u]:
            j = rd.randint(0, n - 1)
        t.append([u, i, j])
    train_batch = np.asarray(t)
    return train_batch


def generate_train_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_1, user_ratings_2, user_ratings_test_2,
                                         n_2, batch_size):
    t_1 = []
    t_2 = []
    for b in range(batch_size):
        u = rd.sample(user_ratings_1.keys(), 1)[0]
        i_1 = rd.sample(user_ratings_1[u], 1)[0]
        i_2 = rd.sample(user_ratings_2[u], 1)[0]
        while i_1 == user_ratings_test_1[u]:
            i_1 = rd.sample(user_ratings_1[u], 1)[0]
        while i_2 == user_ratings_test_2[u]:
            i_2 = rd.sample(user_ratings_2[u], 1)[0]
        j_1 = rd.randint(0, n_1 - 1)
        j_2 = rd.randint(0, n_2 - 1)
        while j_1 in user_ratings_1[u]:
            j_1 = rd.randint(0, n_1 - 1)
        while j_2 in user_ratings_2[u]:
            j_2 = rd.randint(0, n_2 - 1)
        t_1.append([u, i_1, j_1])
        t_2.append([u, i_2, j_2])
    train_batch_1 = np.asarray(t_1)
    train_batch_2 = np.asarray(t_2)
    return train_batch_1, train_batch_2


def generate_test_batch(user_ratings, user_ratings_test, n):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        rated = user_ratings[u]
        for j in range(100):
            k = np.random.randint(0, n)
            while k in rated:
                k = np.random.randint(0, n)
            t.append([u, i, k])
        # print(t)
        yield np.asarray(t)


def generate_test_batch_for_conet(user_ratings_1, user_ratings_test_1, n_1, user_ratings_2, user_ratings_test_2, n_2):
    for u_1 in user_ratings_1.keys():
        t_1 = []
        i = user_ratings_test_1[u_1]
        rated = user_ratings_1[u_1]
        for j in range(99):
            k = np.random.randint(0, n_1)
            while k in rated:
                k = np.random.randint(0, n_1)
            t_1.append([u_1, i, k])
    for u_2 in user_ratings_2.keys():
        t_2 = []
        i = user_ratings_test_2[u_2]
        rated = user_ratings_2[u_2]
        for j in range(99):
            k = np.random.randint(0, n_2)
            while k in rated:
                k = np.random.randint(0, n_2)
            t_2.append([u_2, i, k])
        # print(t)
        yield np.asarray(t_1), np.asarray(t_2)


def generate_test_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_1,
                                        user_ratings_2, user_ratings_test_2, n_2):
    for u in user_ratings_1.keys():
        t_1 = []
        t_2 = []
        i_1 = user_ratings_test_1[u]
        i_2 = user_ratings_test_2[u]
        rated_1 = user_ratings_1[u]
        rated_2 = user_ratings_2[u]
        for j in range(99):
            k = np.random.randint(0, n_1-1)
            while k in rated_1:
                k = np.random.randint(0, n_1-1)
            t_1.append([u, i_1, k])
        for j in range(99):
            k = np.random.randint(0, n_2-1)
            while k in rated_2:
                k = np.random.randint(0, n_2-1)
            t_2.append([u, i_2, k])
        yield np.asarray(t_1), np.asarray(t_2)


def load_data(filepath):
    n_users = 0
    n_items = 0
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) > 0:
                n_users += 1
                l = l.strip('\n').rstrip(' ')
                items = [int(i) for i in l.split(' ')]
                n_items = max(n_items, max(items))

    n_items += 1
    user_ratings = defaultdict(dict)
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]

            user_ratings[uid] = train_items

    return n_users, n_items, user_ratings


def generate_adjacent_matrix(train_file_path, n_users, n_items):
    print("generate adjacent_matrix using data from " + train_file_path)
    R = np.zeros((n_users, n_items), dtype=np.float32)
    with open(train_file_path) as f_train:
        for l in f_train.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]

            for i in train_items:
                R[uid, i] = 1
    return R


def adjacent_matrix(graph, self_connection=False):
    print('begin to calculate adjacent matrix')
    num_user = graph.shape[0]
    num_item = graph.shape[1]
    A = sp.dok_matrix((num_user + num_item, num_user + num_item), dtype=np.float32)
    A[:num_user, num_user:] = graph
    A[num_user:, :num_user] = graph.T

    if self_connection:
        return np.identity(num_user+num_item, dtype=np.float32) + A
    return A.tocsr()


def generate_cross_adjacent_matrix(filepath_1, filepath_2, num_common_users, num_distinct_users_1, num_distinct_users_2, num_items_1, num_items_2, dataset_mode='common_ahead'):
    print('Building cross data adjacent matrix...')
    R_1 = sp.dok_matrix((num_common_users, num_items_1), dtype=np.float32)
    R_2 = sp.dok_matrix((num_common_users, num_items_2), dtype=np.float32)

    if dataset_mode == 'common_ahead':
        with open(filepath_1) as f_1:
            common_ratings_1 = f_1.readlines()[:num_common_users]
        with open(filepath_2) as f_2:
            common_ratings_2 = f_2.readlines()[:num_common_users]
        for l in common_ratings_1:
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                R_1[uid, i] = 1.0

        for l in common_ratings_2:
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                R_2[uid, i] = 1.0
 
    if dataset_mode == 'distinct_ahead':
        with open(filepath_1) as f_1:
            common_ratings_1 = f_1.readlines()[-num_common_users:]
        with open(filepath_2) as f_2:
            common_ratings_2 = f_2.readlines()[-num_common_users:]
        for l in common_ratings_1:
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                R_1[uid-num_distinct_users_1, i] = 1.0

        for l in common_ratings_2:
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                R_2[uid-num_distinct_users_2, i] = 1.0


    R_1, R_2 = R_1.tolil(), R_2.tolil()

    plain_adj_mat = sp.dok_matrix((num_items_1 + num_common_users + num_items_2, num_items_1 + num_common_users + num_items_2),
                                      dtype=np.float32).tolil()
    plain_adj_mat[num_items_1: num_items_1 + num_common_users, :num_items_1] = R_1
    plain_adj_mat[:num_items_1, num_items_1: num_items_1 + num_common_users] = R_1.T
    plain_adj_mat[num_items_1: num_items_1 + num_common_users, num_items_1 + num_common_users:] = R_2
    plain_adj_mat[num_items_1 + num_common_users:, num_items_1: num_items_1 + num_common_users] = R_2.T
    plain_adj_mat = plain_adj_mat.todok()
    norm_adj_mat = normalized_adj_single(plain_adj_mat + sp.eye(plain_adj_mat.shape[0]))
    print('Get adjacent matrix successfully.')
    return norm_adj_mat

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
def normalized_adj_single(adj):
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj


def get_train_instances(filepath, test_rating, num_items):
    train_user_input, train_item_input, train_labels = [], [], []
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                if i == test_rating[uid]:
                    continue
                train_user_input.append(uid)
                train_item_input.append(i)
                train_labels.append(1)
                for _ in range(1):
                    j = np.random.randint(num_items)
                    while j in train_items:
                        j = np.random.randint(num_items)
                    train_user_input.append(uid)
                    train_item_input.append(j)
                    train_labels.append(0)

    return train_user_input, train_item_input, train_labels, len(train_labels)


def get_test_instances(filepath, test_rating, num_items):
    test_user_input, test_item_input, test_labels = [], [], []
    has_interact = defaultdict()
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            has_interact[uid] = train_items
    for u in test_rating.keys():
        test_user_input.append(u)
        test_item_input.append(test_rating[u])
        test_labels.append(1)
        for j in range(99):
            k = np.random.randint(0, num_items-1)
            while k in has_interact[u]:
                k = np.random.randint(0, num_items-1)
            test_user_input.append(u)
            test_item_input.append(k)
            test_labels.append(0)
    return test_user_input, test_item_input, test_labels


def generate_train_instance_for_each_epoch(user_ratings, user_ratings_test, n):
    t = []
    for u in user_ratings.keys():
        for i in user_ratings[u]:
            if i == user_ratings_test[u]:
                continue
            j = rd.randint(0, n-1)
            while j in user_ratings[u]:
                j = rd.randint(0, n - 1)
            t.append([u, i, j])
    train_batch = np.asarray(t)
    print(train_batch.shape)
    return train_batch, len(train_batch)


# generated_sample_1, num_of_sample_1 = generate_train_instance_for_each_epoch(user_ratings_1, user_ratings_test_1,
#                                                                              n_items_1)
# generated_sample_2, num_of_sample_2 = generate_train_instance_for_each_epoch(user_ratings_2, user_ratings_test_2,
#                                                                              n_items_2)
# num_batches = math.ceil(max(num_of_sample_1, num_of_sample_2)/params.BATCH_SIZE)
# sample_ids_1 = [sid for sid in range(num_of_sample_1)]
# sample_ids_2 = [sid for sid in range(num_of_sample_2)]
# rd.shuffle(sample_ids_1)
# rd.shuffle(sample_ids_2)
# for i in range(num_batches):
#     sample_of_this_batch_1 = []
#     sample_of_this_batch_2 = []
#     for b in range(params.BATCH_SIZE):
#         if not sample_ids_1:
#             sample_id_1 = rd.randrange(0, num_of_sample_1)
#         else:
#             sample_id_1 = sample_ids_1.pop()
#         sample_of_this_batch_1.append(generated_sample_1[sample_id_1])
#         if not sample_ids_2:
#             sample_id_2 = rd.randrange(0, num_of_sample_2)
#         else:
#             sample_id_2 = sample_ids_2.pop()
#         sample_of_this_batch_2.append(generated_sample_1[sample_id_1])
#
#     sample_of_this_batch_1 = np.asarray(sample_of_this_batch_1)
#     sample_of_this_batch_2 = np.asarray(sample_of_this_batch_2)
#     print(sample_of_this_batch_1.shape)
#     print(sample_of_this_batch_2.shape)











