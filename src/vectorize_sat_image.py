import os
import shutil
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import gc
import numpy as np
import hickle as hkl

def vectorize_sat_image():
    minimizing_im_sz = 4
    desired_im_sz = (512//minimizing_im_sz, 672//minimizing_im_sz)  # (w, h)
    DATA_DIR = './data/features'


    img_df = pd.read_csv('./data/external/img_path_info.csv')

    nt = 96//4
    cur_loc = 0
    possible_starts = []
    while cur_loc < len(img_df) - nt + 1:
        possible_starts.append(cur_loc)
        cur_loc += nt
    train_starts, val_starts, _, _ = train_test_split(
        possible_starts, possible_starts, test_size=0.2, random_state=42)

    splits = {s: [] for s in ['train', 'val']}
    splits['train'] = train_starts
    splits['val'] = val_starts
    for split in splits:
        im_list = []
        source_list = []
        len_split = len(splits[split])
        for loc in splits[split]:
            img_index = img_df.index.values[loc:loc+nt]
            im_list += [img_df['img_path'][im] for im in img_index]
            source_list += [loc] * nt
        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = []
        for im_file in tqdm(im_list):
            im_file = im_file
            img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=desired_im_sz,
                             interpolation=cv2.INTER_CUBIC)
            img = img.tolist()
            X.append(img)
            del img
            gc.collect()
        X = np.asarray(X)
        # (len, h, w, ch)
        X = X.reshape(len(im_list), desired_im_sz[1], desired_im_sz[0], 1)
        print(X.shape)
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(
            DATA_DIR, 'sources_' + split + '.hkl'))
