import numpy as np
import pandas as pd
import scipy
import os
from sklearn.metrics.pairwise import cosine_similarity


def build_items_neighbour(train):
    iu_dict = {i: train[train[1] == i][0].tolist() for i in train[1].unique().tolist()}
    return iu_dict


dataset = 'baby'
top_k = 10

train_reviews = pd.read_csv(f'./data/{dataset}/5-core/train_reviews.txt', sep='\t', header=None)
item_user_dict = build_items_neighbour(train_reviews)

initial_users = train_reviews[0].unique().tolist()
initial_items = train_reviews[1].unique().tolist()
initial_num_users = len(initial_users)
initial_num_items = len(initial_items)

# public --> private reindexing
public_to_private_users = {u: idx for idx, u in enumerate(initial_users)}
public_to_private_items = {i: idx for idx, i in enumerate(initial_items)}

# private --> public reindexing
private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

rows = [public_to_private_users[u] for u in train_reviews[0].tolist()]
cols = [public_to_private_items[i] for i in train_reviews[1].tolist()]

R = scipy.sparse.csr_matrix(([1] * len(rows), (rows, cols)), shape=(initial_num_users, initial_num_items))
R_I = R.T @ R
item_rows, item_cols = R_I.nonzero()
item_values = R_I.data
max_all_items = item_values.max()

del R, R_I

item_rows_sim = []
item_cols_sim = []
dot_values = []
min_values = []
max_values = []
global_values = []

for idx, (i1, i2, value) in enumerate(zip(item_rows, item_cols, item_values)):
    if (idx + 1) % 1000 == 0:
        print(str(idx + 1) + '/' + str(item_rows.shape[0]))
    if i1 != i2:
        i1_users = item_user_dict[private_to_public_items[i1]]
        i2_users = item_user_dict[private_to_public_items[i2]]
        co_occurred_users = set(i1_users).intersection(set(i2_users))
        num_user_intersection = len(co_occurred_users)
        num_user_union = len(set(i1_users).union(set(i2_users)))
        min_user = min(len(i1_users), len(i2_users))
        max_user = max(len(i1_users), len(i2_users))
        clust_coeff_dot = num_user_intersection / num_user_union
        clust_coeff_min = num_user_intersection / min_user
        clust_coeff_max = num_user_intersection / max_user
        global_coeff = num_user_intersection / max_all_items
        sum_ = 0
        for user in co_occurred_users:
            left = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                        str(private_to_public_items[i1]) + '_' + str(user)) + '.npy')
            right = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                         str(private_to_public_items[i2]) + '_' + str(user)) + '.npy')
            dist = cosine_similarity(left, right)[0, 0]
            if dist > 0.0:
                sum_ += dist
        if num_user_intersection > 0:
            item_rows_sim.append(i1)
            item_cols_sim.append(i2)
            dot_values.append(sum_ * clust_coeff_dot)
            min_values.append(sum_ * clust_coeff_min)
            max_values.append(sum_ * clust_coeff_max)
            global_values.append(sum_ * global_coeff)

R_I = scipy.sparse.csr_matrix((dot_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.csr_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_dot_sparse.npz', R_I)

R_I = scipy.sparse.csr_matrix((min_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.csr_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_min_sparse.npz', R_I)

R_I = scipy.sparse.csr_matrix((max_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.csr_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_max_sparse.npz', R_I)

R_I = scipy.sparse.csr_matrix((global_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.csr_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_global_sparse.npz', R_I)
