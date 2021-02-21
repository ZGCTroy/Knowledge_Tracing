import os

import pandas as pd
import numpy as np



from data_helper import label_encoding, label_transform
from data_helper import setup_pandas
from data_helper import calculate_and_add_correctness_ratio, calculate_extended_information
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
        usecols=['order_id', 'user_id', 'correct', 'skill_name','problem_id','ms_first_response'],
        dtype={
            'correct': 'int',
            'skill_name': 'category'
        },
    )
    print('successfully read the file :', filename)

    # TODO 2 : Dropna , DropDupliates, Sort, ResetIndex
    df = df.dropna().drop_duplicates()
    df = df[df['ms_first_response']>=0]
    df = df.sort_values(by=['user_id', 'order_id','problem_id'])
    df = df.reset_index(drop=True)

    # TODO 3 : LabelTransform: user_id, skill_id, problem_id
    df['user_id'] = label_encoding(df['user_id'])
    df['skill_id'] = label_encoding(df['skill_name'])
    df['problem_id'] = label_encoding(df['problem_id'])

    df = calculate_extended_information(df)

    df['power'] = label_encoding(df['power'])
    df['skill_difficulty'] = label_encoding(df['skill_difficulty'])
    df['problem_difficulty'] = label_encoding(df['problem_difficulty'])

    power_num = len(df['power'].unique())
    skill_difficulty_num = len(df['skill_difficulty'].unique())
    problem_difficulty_num = len(df['problem_difficulty'].unique())
    skill_num = len(df['skill_id'].unique())
    problem_num = len(df['problem_id'].unique())
    print('skill num', skill_num)
    print('problem num', problem_num)
    print('power num', power_num)
    print('skill difficulty num', skill_difficulty_num)
    print('problem difficulty num', problem_difficulty_num)

    print(df['skill_difficulty'].value_counts())
    print(df['problem_difficulty'].value_counts())

    # TODO 4: Groupby: user_id
    df = df[['user_id','skill_id','correct','problem_id','power','skill_difficulty','problem_difficulty']]
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

    df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed.csv'),
        mode='w',
        index=False
    )


    # TODO 6: split train, val, Test
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
        root_dir='../data/Assistment09',
        filename='skill_builder_data_corrected'
    )

    # from PathSim import cal_similarity
    # cal_similarity(
    #     root_dir='../data/Assistment09',
    #     filename = 'skill_builder_data_corrected_preprocessed'
    # )

run()

