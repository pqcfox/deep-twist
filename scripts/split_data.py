#!/usr/bin/env python
import re
import glob
import shutil
import os.path
import numpy as np
from tqdm import tqdm

TRAIN_FRAC = 0.8

os.chdir('cornell')
if 'train' in os.listdir() or 'test' in os.listdir():
    print('Split already created. Aborting.')
else:
    os.mkdir('train')
    os.mkdir('test')
    rgbs = glob.glob('pcd*r.png')
    ids = [re.findall('\d+', fname)[0] for fname in rgbs]
    train_count = int(len(ids) * TRAIN_FRAC)
    train_ids = np.random.choice(ids, train_count, replace=False)
    test_ids = [fid for fid in ids if fid not in train_ids]

    for dest, ids in [('train', train_ids), ('test', test_ids)]:
        print('Copying {} files...'.format(dest))
        for fid in tqdm(ids):
            for fname in glob.glob('*{}*.*'.format(fid)):
                shutil.copy(fname, dest)
