import os

import pandas as pd
import numpy as np



from data_helper import label_encoding, label_transform
from data_helper import setup_pandas
from data_helper import calculate_and_add_correctness_ratio
from data_helper import train_val_test_split
from data_helper import get_one_user_data
from data_helper import print_info
from data_helper import setup_seed


def pre_process(root_dir, filename):
    # TODO 1: Read
    print('start to process the file : ',filename)
    df = pd.read_csv(
        os.path.join(root_dir, filename) + '.csv',
        header=0,
        usecols=['startTime', 'studentId', 'correct', 'skill','problemId'],
    )
    print('successfully read the file :', filename)

    # TODO 2 : Dropna , DropDupliates, Sort, ResetIndex
    df = df.dropna().drop_duplicates()
    df = df.sort_values(by=['studentId', 'startTime'])
    df = df.reset_index(drop=True)

    # TODO 3 : LabelTransform: user_id, skill_id, problem_id
    df['user_id'] = label_encoding(df['studentId'])
    df['skill_id'] = label_encoding(df['skill'])
    df['problem_id'] = label_encoding(df['problemId'])
    skill_num = len(df['skill_id'].unique())
    problem_num = len(df['problem_id'].unique())
    print('skill num',skill_num)
    print('problem num',problem_num)

    df = df[['user_id','skill_id','correct','problem_id']]

    # TODO 4: Groupby: user_id
    grouped_df = df.groupby('user_id')
    users_list = list(grouped_df.groups.keys())
    df = pd.DataFrame()

    for user_id in users_list:
        user_df = grouped_df.get_group(user_id)
        if (len(user_df)) <= 15:
            continue

        # 一个user_id 做成一个 list str 数据
        user_df = get_one_user_data(user_df, skill_num=skill_num, mode='skill states on all')

        # concat
        df = pd.concat([df, user_df])

    print('start to save data in csv file')
    print(df.head(10))
    df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed.csv'),
        mode='w',
        index=False
    )


    # TODO 6: split train, val, test
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    # split
    train_df, val_df, test_df = train_val_test_split(
        df,
        train_test_ratio=0.7,
        train_val_ratio=0.8
    )

    train_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_train.csv'),
        mode='w',
        index=False
    )

    val_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_val.csv'),
        mode='w',
        index=False
    )

    test_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_test.csv'),
        mode='w',
        index=False
    )

    print('finish to save data in csv file\n\n')


def run():
    setup_seed(41)
    setup_pandas()

    pre_process(
        root_dir='../data/Assistment17',
        filename='anonymized_full_release_competition_dataset'
    )

run()

