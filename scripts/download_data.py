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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)

    print('Pouring data into root folder...')
    for i in tqdm(range(1, 11)):
        dirname = '{:02d}'.format(i)
        for fname in glob.glob(os.path.join(dirname, '*.*')):
            shutil.move(fname, '.')
        os.rmdir(dirname)
else:
    print('Data folder not empty. Aborting.')
