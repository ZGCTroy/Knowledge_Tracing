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


def label_transform(df, col_name):
    label_encoder = LabelEncoder()
    df[col_name] = label_encoder.fit_transform(df[col_name]) + 1
    mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print(mapping)
    return df

def category_transform(serie):
    # serie = pd.cut(x=serie, bins=[-1,0,1,2,3,10000], labels=[0,1,2,3,4])
    def mapping(x):
        if x >= 4:
            return 4 + 1
        else:
            return x + 1

    serie = serie.map(mapping)

    return serie

def calculate_and_add_correctness_ratio(df):
    grouped_df = df.groupby(by=['user_id','skill_id'])
    df['same_skill_total_num'] = grouped_df['correct'].transform(len)
    df['same_skill_correct_num'] = grouped_df['correct'].transform(sum)
    df['same_skill_correctness_ratio'] = df['same_skill_correct_num'] / df['same_skill_total_num']

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

    df = label_transform(df, 'user_id')
    df = label_transform(df, 'skill_id')
    df = label_transform(df, 'problem_id')

    df['attempt_count'] = category_transform(df['attempt_count'])

    grouped_df = df.groupby('user_id')
    users_list = list(grouped_df.groups.keys())

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for user_id in users_list:
        user_df = grouped_df.get_group(user_id)
        if (len(user_df)) < 30:
            continue
        train_len = int(len(user_df) * 0.7)
        temp_test_df = user_df.iloc[train_len:]
        temp_train_df = user_df.iloc[0:train_len]
        train_len = int(train_len * 0.8)
        temp_val_df = temp_train_df.iloc[train_len:]
        temp_train_df = temp_train_df.iloc[0:train_len]

        # 计算 skill正确率
        temp_train_df = calculate_and_add_correctness_ratio(temp_train_df)

        train_df = pd.concat([train_df, temp_train_df])
        test_df = pd.concat([test_df, temp_test_df])
        val_df = pd.concat([val_df, temp_val_df])




    df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed.csv'),
        mode='w',
        index=0
    )
    train_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_train.csv'),
        mode='w',
        index=0
    )
    test_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_test.csv'),
        mode='w',
        index=0
    )
    val_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_val.csv'),
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
