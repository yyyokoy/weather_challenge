import os
import glob
import pandas as pd
import shutil
from datetime import datetime as dt
import datetime

import pandas as pd
import cv2
from tqdm import tqdm
import gc
import numpy as np
import hickle as hkl

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# params
SAT_DIR = './data/input/test/sat'
EXTERNAL_DIR = './data/external'

test_info = './data/input/inference_terms.csv'
test_info_df = pd.read_csv(test_info)
test_info_df.head()
# ==============================================================================


# ==============================================================================
def create_test_img_path_df():

    img_list = glob.glob(os.path.join(SAT_DIR, '*/*'))
    img_df = pd.DataFrame({'img_path': img_list})
    img_df['img_file_name'] = img_df['img_path'].apply(
        lambda x: x.split('/')[-1])
    img_df['img_date'] = img_df['img_file_name'].apply(
        lambda x: x.split('.')[0])
    img_df['img_date'] = pd.to_datetime(img_df['img_date'])
    img_df.index = img_df['img_date']
    start_at, end_at = img_df['img_date'].min(), img_df['img_date'].max()
    img_df = img_df.reindex(pd.date_range(start_at, end_at, freq="H"))
    img_not_exist_df = img_df[img_df['img_date'].isna()]

    if len(img_df[img_df['img_date'].isna()]) == 0:
        img_not_exist_df = img_df[img_df['img_file_name'].str.contains(
            'dummy')]
        img_df = img_df[~img_df['img_file_name'].str.contains('dummy')]

    img_not_exist_df['img_date'] = img_not_exist_df.index.values
    img_not_exist_df['img_date'] = img_not_exist_df['img_date'].dt.strftime(
        '%Y-%m-%d-%H-%M')
    img_not_exist_df['img_file_name'] = img_not_exist_df['img_date'] + \
        '.dummy.fv.png'
    img_not_exist_df['img_year'] = img_not_exist_df['img_file_name'].apply(
        lambda x: x.split('-')[0])
    img_not_exist_df['img_month'] = img_not_exist_df['img_file_name'].apply(
        lambda x: x.split('-')[1])
    img_not_exist_df['img_day'] = img_not_exist_df['img_file_name'].apply(
        lambda x: x.split('-')[2])
    img_not_exist_df['img_yyyy-mm-dd'] = img_not_exist_df['img_year'] + \
        '-'+img_not_exist_df['img_month'] + '-' + img_not_exist_df['img_day']
    img_not_exist_df['img_path'] = SAT_DIR + '/' + \
        img_not_exist_df['img_yyyy-mm-dd'] + \
        '/' + img_not_exist_df['img_file_name']
    img_not_exist_df = img_not_exist_df[['img_path', 'img_file_name']]
    img_not_exist_df = img_not_exist_df.assign(file_not_exist=1)
    print('number of missing data: {}'.format(len(img_not_exist_df)))
    img_df = img_df.drop('img_date', axis=1)
    img_df = img_df.dropna()
    img_df = img_df.assign(file_not_exist=0)

    img_df = pd.concat([img_df, img_not_exist_df]).sort_index()
    img_df = img_df.reset_index(drop=True)

    if not os.path.exists(EXTERNAL_DIR):
        os.mkdir(EXTERNAL_DIR)
    img_df.to_csv(os.path.join(EXTERNAL_DIR, 'test_img_path_info.csv'))
# ==============================================================================


# ==============================================================================
def vectorize_test_img():
    img_df = pd.read_csv(os.path.join(EXTERNAL_DIR, 'test_img_path_info.csv'))
    for index, row in img_df[img_df['file_not_exist'] == 1].iterrows():
        base_dir = os.path.split(row.img_path)[0]
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        try:
            shutil.copyfile(img_df.iloc[index-1].img_path, row.img_path)
        except FileNotFoundError:
            pass

    all_filepaths = []
    for end_date in test_info_df.OpenData_96hr_End:
        filepaths = []
        end_date = dt.strptime(end_date, '%Y/%m/%d %H:%M')
        dir_name = end_date.strftime('%Y-%m-%d')
        filename = end_date.strftime('%Y-%m-%d-%H-%M')
        filepath = f'./data/input/test/sat/{dir_name}/{filename}.fv.png'
        filepaths.append(filepath)
        for t in range(1,24):

            date = end_date - datetime.timedelta(hours=t)
            dir_name = date.strftime('%Y-%m-%d')
            filename = date.strftime('%Y-%m-%d-%H-%M')

            filepath = f'./data/input/test/sat/{dir_name}/{filename}.fv.png'
            filepaths.append(filepath)
            filepaths = sorted(filepaths)
        for _ in range(0,24):
            filepaths.append('none')
        all_filepaths += filepaths

    minimizing_im_sz = 4
    desired_im_sz = (512//minimizing_im_sz, 672//minimizing_im_sz)  # (w, h)
    DATA_DIR = './data/features'

    nt = 48
    cur_loc = 0
    test_starts = []
    while cur_loc < len(all_filepaths) - nt + 1:
        test_starts.append(cur_loc)
        cur_loc += nt
    test_starts
    im_list = []
    source_list = []
    for loc in test_starts:
        source_list += [loc] * nt
    print('Creating test data: ' + str(len(filepaths)) + ' images')
    X = []
    for im_file in tqdm(all_filepaths):
        if im_file == 'none':
            img = np.zeros(desired_im_sz)
        else:
            img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
        try:
            img = cv2.resize(img, dsize=desired_im_sz,
                             interpolation=cv2.INTER_CUBIC)
        except:
            split1 = os.path.splitext(im_file)
            split2 = os.path.splitext(split1[0])
            dummy_file = split2[0] + '.dummy' + split2[1] + split1[1]
            img = cv2.imread(dummy_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=desired_im_sz,
                     interpolation=cv2.INTER_CUBIC)
        img = img.tolist()
        X.append(img)
        del img
        gc.collect()
    X = np.asarray(X)
    # (len, h, w, ch)
    X = X.reshape(len(all_filepaths), desired_im_sz[1], desired_im_sz[0], 1) # (len, h, w, ch)
    print(X.shape)
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    hkl.dump(X, os.path.join(DATA_DIR, 'X_test.hkl'))
    hkl.dump(source_list, os.path.join(
        DATA_DIR, 'sources_test.hkl'))
# ==============================================================================


if __name__ == '__main__':
    create_test_img_path_df()
    vectorize_test_img()