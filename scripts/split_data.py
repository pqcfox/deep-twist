#!/usr/bin/env python
import re
import glob
import shutil
import os.path
import numpy as np
from tqdm import tqdm

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

os.chdir('cornell')
if 'train' in os.listdir() or 'test' in os.listdir():
    print('Split already created. Aborting.')
else:
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('val')
    rgbs = glob.glob('pcd*r.png')
    ids = np.array([re.findall('\d+', fname)[0] for fname in rgbs])
    indexes = np.arange(len(ids))
    np.random.shuffle(indexes)
    train_count = int(len(ids) * TRAIN_FRAC)
    val_count = int(len(ids) * VAL_FRAC)
    train_ids = ids[indexes[:train_count]]
    val_ids = ids[indexes[train_count:(train_count + val_count)]]
    test_ids = ids[indexes[(train_count + val_count):]]

    for dest, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        print('Copying {} files...'.format(dest))
        for fid in tqdm(ids):
            for fname in glob.glob('*{}*.*'.format(fid)):
                shutil.copy(fname, dest)
