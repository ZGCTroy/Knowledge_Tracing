import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 显示所有列
pd.set_option('display.max_columns', 200)

# 显示所有行
pd.set_option('display.max_rows', None)

# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)
pd.set_option('expand_frame_repr', False)


def transform(df, col_name):
    label_encoder = LabelEncoder()
    df[col_name] = label_encoder.fit_transform(df[col_name]) + 1
    mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print(mapping)
    return df


def pre_process(root_dir, filename):
    df = pd.read_csv(
        os.path.join(root_dir, filename) + '.csv',
        dtype={'skill_name': 'str'},
        usecols=['user_id', 'assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id',
                 'skill_name','attempt_count'],
    )

    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'attempt_count','assistment_id', 'skill_name']]
    df = df.dropna().drop_duplicates()
    df = df.sort_values(by=['user_id', 'order_id', 'problem_id'])
    df = df.reset_index()

    print('Before : ')
    print(df.head(5))
    print()
    df = transform(df, 'user_id')
    df = transform(df, 'skill_id')
    df = transform(df, 'problem_id')
    print('After: ')
    print(df.head(5))
    print()
    print()

    df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed.csv'),
        mode='w',
        index=0
    )

def run():
    pre_process(
        root_dir='data/',
        filename='skill_builder_data_corrected'
    )
    pre_process(
        root_dir='data/',
        filename='skill_builder_data_corrected_small'
    )
    pre_process(
        root_dir='data/',
        filename='skill_builder_data_corrected_big'
    )

run()
