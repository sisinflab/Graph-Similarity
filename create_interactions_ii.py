import numpy as np
import pandas as pd
import scipy
import os
from sklearn.metrics.pairwise import cosine_similarity


def build_items_neighbour(df):
    iu_dict = {i: df[df[1] == i][0].tolist() for i in df[1].unique().tolist()}
    return iu_dict


def dataframe_to_dict(data):
    ratings = data.set_index(0)[[1, 2]].apply(lambda x: (x[1], float(x[2])), 1) \
        .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
    return ratings


dataset = 'baby'
top_k = 10

train = pd.read_csv(f'./data/{dataset}/5-core/train.txt', sep='\t', header=None)
item_user_dict = build_items_neighbour(train)

train_dict = dataframe_to_dict(train)
users = list(train_dict.keys())
items = list({k for a in train_dict.values() for k in a.keys()})

initial_num_users = len(train[0].nunique())
initial_num_items = len(train[1].nunique())

private_to_public_users = {p: u for p, u in enumerate(users)}
public_to_private_users = {v: k for k, v in private_to_public_users.items()}
private_to_public_items = {p: i for p, i in enumerate(items)}
public_to_private_items = {v: k for k, v in private_to_public_items.items()}

rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]

R = scipy.sparse.coo_matrix(([1] * len(rows), (rows, cols)), shape=(initial_num_users, initial_num_items))
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

R_I = scipy.sparse.coo_matrix((dot_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.coo_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_dot_{top_k}.npz', R_I)

R_I = scipy.sparse.coo_matrix((min_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.coo_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_min_{top_k}.npz', R_I)

R_I = scipy.sparse.coo_matrix((max_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.coo_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_max_{top_k}.npz', R_I)

R_I = scipy.sparse.coo_matrix((global_values, (item_rows_sim, item_cols_sim)),
                              shape=(initial_num_items, initial_num_items)).todense()
indices_one = np.argsort(-R_I)[:, :top_k]
indices_zero = np.argsort(-R_I)[:, top_k:]
R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
R_I = scipy.sparse.coo_matrix(R_I)
scipy.sparse.save_npz(f'./data/{dataset}/5-core/ii_global_{top_k}.npz', R_I)
