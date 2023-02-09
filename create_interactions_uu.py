import numpy as np
import pandas as pd
import scipy
import os
from sklearn.metrics.pairwise import cosine_similarity


def build_users_neighbour(train):
    ui_dict = {u: train[train[0] == u][1].tolist() for u in train[0].unique().tolist()}
    return ui_dict


dataset = 'baby'
top_k = 10

train_reviews = pd.read_csv(f'./data/{dataset}/5-core/train_prova.txt', sep='\t', header=None)
user_item_dict = build_users_neighbour(train_reviews)

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
R_U = R @ R.T
user_rows, user_cols = R_U.nonzero()
user_values = R_U.data
max_all_users = user_values.max()

del R, R_U

user_rows_sim = []
user_cols_sim = []
dot_values = []
min_values = []
max_values = []
global_values = []

for idx, (u1, u2, value) in enumerate(zip(user_rows, user_cols, user_values)):
    if (idx + 1) % 1000 == 0:
        print(str(idx + 1) + '/' + str(user_rows.shape[0]))
    if u1 != u2:
        u1_items = user_item_dict[private_to_public_users[u1]]
        u2_items = user_item_dict[private_to_public_users[u2]]
        co_occurred_items = set(u1_items).intersection(set(u2_items))
        num_item_intersection = len(co_occurred_items)
        num_item_union = len(set(u1_items).union(set(u2_items)))
        min_item = min(len(u1_items), len(u2_items))
        max_item = max(len(u1_items), len(u2_items))
        clust_coeff_dot = num_item_intersection / num_item_union
        clust_coeff_min = num_item_intersection / min_item
        clust_coeff_max = num_item_intersection / max_item
        global_coeff = num_item_intersection / max_all_users
        sum_ = 0
        for item in co_occurred_items:
            left = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                        str(item) + '_' + str(private_to_public_users[u1])) + '.npy')
            right = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                         str(item) + '_' + str(private_to_public_users[u2])) + '.npy')
            dist = cosine_similarity(left, right)[0, 0]
            if dist > 0.0:
                sum_ += dist
        if num_item_intersection > 0:
            user_rows_sim.append(u1)
            user_cols_sim.append(u2)
            dot_values.append(sum_ * clust_coeff_dot)
            min_values.append(sum_ * clust_coeff_min)
            max_values.append(sum_ * clust_coeff_max)
            global_values.append(sum_ * global_coeff)

R_U = scipy.sparse.csr_matrix((dot_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.csr_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_dot_sparse.npz', R_U)

R_U = scipy.sparse.csr_matrix((min_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.csr_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_min_sparse.npz', R_U)

R_U = scipy.sparse.csr_matrix((max_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.csr_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_max_sparse.npz', R_U)

R_U = scipy.sparse.csr_matrix((global_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.csr_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_global_sparse.npz', R_U)
