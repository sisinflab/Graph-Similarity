import itertools
import json
import numpy as np
import pandas as pd
import os
from operator import itemgetter
from scipy.spatial import distance

train_reviews = pd.read_csv('train_reviews.txt', sep='\t', header=None)

def build_users_neighbour():
    ui_dict = {u: train_reviews[train_reviews[0] == u][1].tolist() for u in train_reviews[0].unique().tolist()}
    return ui_dict

users_neighbour = build_users_neighbour()
users = train_reviews[0].unique().tolist()
user_user = list(itertools.combinations(users, 2))
user_a, user_b = list(map(list, zip(*user_user)))
dict_a = list(itemgetter(*user_a)(users_neighbour))
dict_b = list(itemgetter(*user_b)(users_neighbour))
out = list(map(lambda x: list(set(x[0]).intersection(set(x[1]))), zip(dict_a, dict_b)))
out = list(filter(lambda x: len(x[2]) > 0, zip(user_a, user_b, out)))
avg_interactions = [len(users) for _, _, users in out]
avg_interactions = sum(avg_interactions) / len(avg_interactions)
out_dict = dict()
out_dict_sim = dict()

#if not os.path.exists('./user_user/'):
#    os.makedirs('./user_user/')

for u1, u2, items in out:
    list_of_interaction_pairs = [[str(i) + '_' + str(u1), str(i) + '_' + str(u2)] for i in items]
    out_dict[str(u1) + '_' + str(u2)] = list_of_interaction_pairs
    npy = np.empty((len(list_of_interaction_pairs), 2, 768))
    avg = 0
    for idx, interaction_pair in enumerate(list_of_interaction_pairs):
        npy[idx, 0, :] = np.load(os.path.join('./reviews/', interaction_pair[0]) + '.npy')
        npy[idx, 1, :] = np.load(os.path.join('./reviews/', interaction_pair[1]) + '.npy')
        dist = distance.cosine(npy[idx, 1, :], npy[idx, 0, :])
        dist = dist / 2
        avg += dist
    avg /= len(list_of_interaction_pairs)
    out_dict_sim[str(u1) + '_' + str(u2)] = avg * (len(interaction_pair) / avg_interactions)
        
#    np.save(os.path.join('./user_user/', str(u1) + '_' + str(u2)) + '.npy', npy)

with open('interactions_uu.json', 'w') as f:
    json.dump(out_dict, f)

with open('interactions_sim_uu.json', 'w') as f:
    json.dump(out_dict_sim, f)
