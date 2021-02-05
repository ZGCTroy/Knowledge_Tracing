import os

import pandas as pd



from data_helper import label_encoding, label_transform
from data_helper import setup_pandas
from data_helper import calculate_and_add_correctness_ratio
from data_helper import train_val_test_split
from data_helper import get_one_user_data
from data_helper import print_info


def pre_process(root_dir, filename):
    # TODO 1: Read
    print('start to process the file : ',filename)
    df = pd.read_csv(
        os.path.join(root_dir, filename) + '.csv',
        header=0,
        usecols=['log_id', 'user_id', 'correct', 'sequence_id'],
    )
    print('successfully read the file :', filename)

    # TODO 2 : Dropna , DropDupliates, Sort, ResetIndex
    # TODO 3 : LabelTransform: user_id, skill_id, problem_id
    df = df.dropna().drop_duplicates()
    df = df.sort_values(by=['user_id', 'log_id'])
    df = df.reset_index()
    df['user_id'] = label_encoding(df['user_id'])
    df['skill_id'] = label_encoding(df['sequence_id'])
    skill_num = len(df['skill_id'].unique())
    df.loc[df['correct'] >= 0.5,'correct'] = 1
    df.loc[df['correct'] < 0.5, 'correct'] = 0
    df['correct'] = df['correct'].astype(int)

    df = df[['user_id','skill_id','correct']]


    # TODO 5: Groupby: user_id
    grouped_df = df.groupby('user_id')
    users_list = list(grouped_df.groups.keys())

    df = pd.DataFrame()
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    cnt = 0
    for user_id in users_list:
        user_df = grouped_df.get_group(user_id)
        if (len(user_df)) <= 15:
            continue

        # train val test split
        user_train_df, user_val_df, user_test_df = train_val_test_split(
            user_df,
            train_test_ratio=0.7,
            train_val_ratio=0.8
        )

        # 计算 skill正确率
        user_df = calculate_and_add_correctness_ratio(user_df)
        user_train_df = calculate_and_add_correctness_ratio(user_train_df)
        user_val_df = calculate_and_add_correctness_ratio(user_val_df)
        user_test_df = calculate_and_add_correctness_ratio(user_test_df)

        # 一个user_id 做成一个 list str 数据
        user_df = get_one_user_data(user_df, skill_num=skill_num)
        user_train_df = get_one_user_data(user_train_df,skill_num=skill_num)
        user_val_df = get_one_user_data(user_val_df,skill_num=skill_num)
        user_test_df = get_one_user_data(user_test_df,skill_num=skill_num)

        df = pd.concat([df, user_df])
        train_df = pd.concat([train_df, user_train_df])
        test_df = pd.concat([test_df, user_test_df])
        val_df = pd.concat([val_df, user_val_df])

    # TODO 6: save the csv
    print('start to save data in csv file')

    df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed.csv'),
        mode='w',
        index=False
    )

    train_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_train.csv'),
        mode='w',
        index=False
    )
    test_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_test.csv'),
        mode='w',
        index=False
    )
    val_df.to_csv(
        os.path.join(root_dir, filename + '_preprocessed_val.csv'),
        mode='w',
        index=False
    )
    print('finish to save data in csv file\n\n')


def run():
    setup_pandas()

    pre_process(
        root_dir='data/Assistment15',
        filename='2015_100_skill_builders_main_problems'
    )

run()
