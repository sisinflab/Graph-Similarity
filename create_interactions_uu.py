import numpy as np
import pandas as pd
import scipy
import os
from sklearn.metrics.pairwise import cosine_similarity


def build_users_neighbour(df):
    ui_dict = {u: df[df[0] == u][1].tolist() for u in df[0].unique().tolist()}
    return ui_dict


def dataframe_to_dict(data):
    ratings = data.set_index(0)[[1, 2]].apply(lambda x: (x[1], float(x[2])), 1) \
        .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
    return ratings


dataset = 'baby'
top_k = 10

train = pd.read_csv(f'./data/{dataset}/5-core/train.txt', sep='\t', header=None)
user_item_dict = build_users_neighbour(train)

train_dict = dataframe_to_dict(train)
users = list(train_dict.keys())
items = list({k for a in train_dict.values() for k in a.keys()})

initial_num_users = train[0].nunique()
initial_num_items = train[1].nunique()

private_to_public_users = {p: u for p, u in enumerate(users)}
public_to_private_users = {v: k for k, v in private_to_public_users.items()}
private_to_public_items = {p: i for p, i in enumerate(items)}
public_to_private_items = {v: k for k, v in private_to_public_items.items()}

rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]

R = scipy.sparse.coo_matrix(([1] * len(rows), (rows, cols)), shape=(initial_num_users, initial_num_items))
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

R_U = scipy.sparse.coo_matrix((dot_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.coo_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_dot_sparse.npz', R_U)

R_U = scipy.sparse.coo_matrix((min_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.coo_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_min_sparse.npz', R_U)

R_U = scipy.sparse.coo_matrix((max_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.coo_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_max_sparse.npz', R_U)

R_U = scipy.sparse.coo_matrix((global_values, (user_rows_sim, user_cols_sim)),
                              shape=(initial_num_users, initial_num_users)).todense()
indices_one = np.argsort(-R_U)[:, :top_k]
indices_zero = np.argsort(-R_U)[:, top_k:]
R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
R_U = scipy.sparse.coo_matrix(R_U)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/uu_global_sparse.npz', R_U)
