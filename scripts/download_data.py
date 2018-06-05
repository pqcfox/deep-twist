#!/usr/bin/env python
import glob
import gzip
import os.path
import tarfile
import shutil
import urllib.request
from tqdm import tqdm

fnames = [fname for fname in os.listdir('cornell') if fname != '.gitkeep']
if len(fnames) == 0:
    os.chdir('cornell')
    url = 'http://pr.cs.cornell.edu/grasping/rect_data/temp/data{:02d}.tar.gz'

    fnames = []
    print('Downloading data...')
    for i in tqdm(range(1, 11)):
        formatted_url = url.format(i)
        fname = formatted_url.rsplit('/', 1)[1]
        fnames.append(fname)
        urllib.request.urlretrieve(formatted_url, fname)
    
    print('Unpacking data...')
    for i in tqdm(range(1, 11)):
        with tarfile.open(fnames[i-1], 'r:gz') as tar:
            tar.extractall() 

    print('Pouring data into root folder...')
    for i in tqdm(range(1, 11)):
        dirname = '{:02d}'.format(i)
        for fname in glob.glob(os.path.join(dirname, '*.*')):
            shutil.move(fname, '.')
        os.rmdir(dirname)
else:
    print('Data folder not empty. Aborting.')
