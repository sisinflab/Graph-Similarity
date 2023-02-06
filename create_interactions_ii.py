import itertools
import json
import numpy as np
import pandas as pd
import os
from operator import itemgetter
from scipy.spatial import distance

def build_items_neighbour(train_reviews):
    iu_dict = {i: train_reviews[train_reviews[1] == i][0].tolist() for i in train_reviews[1].unique().tolist()}
    return iu_dict

train_reviews = pd.read_csv('train_reviews.txt', sep='\t', header=None)
items_neighbour = build_items_neighbour(train_reviews)
items = train_reviews[1].unique().tolist()
item_item = list(itertools.combinations(items, 2))
item_a, item_b = list(map(list, zip(*item_item)))
dict_a = list(itemgetter(*item_a)(items_neighbour))
dict_b = list(itemgetter(*item_b)(items_neighbour))
out = list(map(lambda x: list(set(x[0]).intersection(set(x[1]))), zip(dict_a, dict_b)))
out = list(filter(lambda x: len(x[2]) > 0, zip(item_a, item_b, out)))
avg_interactions = [len(users) for _, _, users in out]
avg_interactions = sum(avg_interactions) / len(avg_interactions)
out_dict = dict()
out_dict_sim = dict()

#if not os.path.exists('./item_item/'):
#    os.makedirs('./item_item/')

for i1, i2, users in out:
    list_of_interaction_pairs = [[str(i1) + '_' + str(u), str(i2) + '_' + str(u)] for u in users]
    out_dict[str(i1) + '_' + str(i2)] = list_of_interaction_pairs
    npy = np.empty((len(list_of_interaction_pairs), 2, 768))
    avg = 0
    for idx, interaction_pair in enumerate(list_of_interaction_pairs):
        npy[idx, 0, :] = np.load(os.path.join('./reviews/', interaction_pair[0]) + '.npy')
        npy[idx, 1, :] = np.load(os.path.join('./reviews/', interaction_pair[1]) + '.npy')
        dist = distance.cosine(npy[idx, 0, :], npy[idx, 1, :])
        dist = dist / 2
        avg += dist
    avg /= len(list_of_interaction_pairs)
    out_dict_sim[str(i1) + '_' + str(i2)] = avg * (len(interaction_pair) / avg_interactions)
        
#    np.save(os.path.join('./item_item/', str(i1) + '_' + str(i2)) + '.npy', npy)

with open('interactions_ii.json', 'w') as f:
    json.dump(out_dict, f)

with open('interactions_sim_ii.json', 'w') as f:
    json.dump(out_dict_sim, f)
