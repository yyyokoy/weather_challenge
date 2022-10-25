import os
import glob
import pandas as pd
import shutil

SAT_DIR = './data/input/train/sat'
EXTERNAL_DIR = './data/external'


def create_data_df():
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

    # 欠損データ確認
    img_not_exist_df = img_df[img_df['img_date'].isna()]

    # すでに欠損データをダミーで置き換えている場合
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

    img_df.to_csv(os.path.join(EXTERNAL_DIR, 'img_path_info.csv'))

    
def create_dummy_image():
    # 直前のデータをコピーしてダミー作成
    img_df = pd.read_csv(os.path.join(EXTERNAL_DIR, 'img_path_info.csv'))
    for index, row in img_df[img_df['file_not_exist'] == 1].iterrows():
        shutil.copyfile(img_df.iloc[index-1].img_path, row.img_path)


if __name__ == '__main__':
    create_data_df()
    create_dummy_image()