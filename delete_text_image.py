import pandas as pd
import os
import numpy as np

folder = './data/baby/'

train = pd.read_csv(folder + '5-core/train.txt', sep='\t', header=None)

itemSet = set(train[1].unique().tolist())

image_feat = np.load(folder + 'image_feat.npy')
text_feat = np.load(folder + 'text_feat.npy')

if not os.path.exists(folder + 'image_feat/'):
    os.makedirs(folder + 'image_feat/')

if not os.path.exists(folder + 'text_feat/'):
    os.makedirs(folder + 'text_feat/')

for idx, row in enumerate(image_feat):
    np.save(folder + 'image_feat/' + str(idx) + '.npy', row)

for idx, row in enumerate(text_feat):
    np.save(folder + 'text_feat/' + str(idx) + '.npy', row)

for f in ['text_feat', 'image_feat']:
    for file in os.listdir(folder + f'{f}/'):
        if int(file.split('.')[0]) not in itemSet:
            os.remove(folder + f'{f}/' + file)
            print(f'Removed_{file}')
