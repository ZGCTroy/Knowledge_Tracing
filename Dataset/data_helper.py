import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import os
import numpy as np
import random

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_pandas():
    # 显示所有列
    pd.set_option('display.max_columns', 200)

    # 显示所有行
    pd.set_option('display.max_rows', None)

    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 1000)
    pd.set_option('expand_frame_repr', False)

def label_transform(df, col_name):
    print('start to LabelTransform {}'.format(col_name))
    label_encoder = LabelEncoder()
    df[col_name] = label_encoder.fit_transform(df[col_name]) + 1
    mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print(mapping)
    print('finish to LabelTransform {}'.format(col_name))
    return df

def label_encoding(serie):
    label_encoder = LabelEncoder()
    new_serie = pd.Series(label_encoder.fit_transform(serie) + 1, dtype='category')
    mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(mapping)

    return new_serie

def calculate_and_add_correctness_ratio(df):
    df = pd.DataFrame(df)

    grouped_df = df.groupby(by=['user_id', 'skill_id'])

    same_skill_total_num = grouped_df['correct'].transform(len)
    same_skill_correct_num = grouped_df['correct'].transform(sum)
    df['same_skill_correctness_ratio'] = same_skill_correct_num.astype('float') / same_skill_total_num.astype('float')

    return df

def train_val_test_split(user_df, train_test_ratio=0.7, train_val_ratio=0.8):

    train_len = int(len(user_df) * train_test_ratio)
    test_df = user_df.iloc[train_len:]
    train_df = user_df.iloc[0:train_len]

    train_len = int(train_len * train_val_ratio)
    val_df = train_df.iloc[train_len:]
    train_df = train_df.iloc[0:train_len]

    return train_df, val_df, test_df

def get_one_user_data(df, skill_num=110, mode='skill states on all'):
    seq_len = len(list(df['user_id']))

    SK = [0.5 for i in range(0,skill_num+1)]
    SK[0] = 0
    SK_total_num = [0 for i in range(0,skill_num+1)]
    SK_correct_num = [0 for i in range(0, skill_num + 1)]
    SK_mask = [0 for i in range(0,skill_num+1)]

    if mode=='skill states on train':
        pos = max(0, int(seq_len * 0.6) - 1)
    elif mode == 'skill states on all':
        pos = seq_len - 1

    for index in range(pos+1):
        skill_id = int(df['skill_id'].iloc[index])
        correct = df['correct'].iloc[index]
        SK_total_num[skill_id] += 1
        if correct == 1:
            SK_correct_num[skill_id] += 1
        SK[skill_id] = float(SK_correct_num[skill_id]) / float(SK_total_num[skill_id])
        SK_mask[skill_id] = 1

    df = pd.DataFrame(
        data={
            'user_id': [int(df['user_id'].iloc[0])],
            'seq_len': [seq_len],
            'skill_id_sequence': [str(list(df['skill_id']))],
            'problem_id_sequence': [str(list(df['problem_id']))],
            'correctness_sequence': [str(list(df['correct']))],
            'skill_states': [str(SK)],
            'skill_states_mask': [str(SK_mask)]
        }
    )

    return df


def print_info(df):
    print()
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe(include='all'))
    print()
    print()