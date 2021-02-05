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
    print('start to LabelTransform {}'.format(col_name))
    label_encoder = LabelEncoder()
    df[col_name] = label_encoder.fit_transform(df[col_name]) + 1
    mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print(mapping)
    print('finish to LabelTransform {}'.format(col_name))
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
    df = pd.DataFrame(df)
    grouped_df = df.groupby(by=['user_id', 'skill_id'])
    df['same_skill_total_num'] = grouped_df['correct'].transform(len)
    df['same_skill_correct_num'] = grouped_df['correct'].transform(sum)
    df['same_skill_correctness_ratio'] = df['same_skill_correct_num'] / df['same_skill_total_num']

    return df


# 将同属于一个user id的多行pandas 数据 以一个list 然后存储为str的方式 做成pandas的一行数据
def sequence_data_augment(sequence, get_len = False, step_size=1):
    cur_len = len(sequence)
    sequences = [str(sequence[0:i + 1]) for i in range(0, cur_len, step_size)]
    seqs_len = []
    if get_len:
        seqs_len = [len(sequence[0:i + 1]) for i in range(0, cur_len, step_size)]
    return sequences, seqs_len


def get_list_df(df, step_size):
    question_sequences, seqs_len = sequence_data_augment(list(df['problem_id']),get_len=True,step_size=step_size)
    user_id = list(df['user_id'])[0]
    user_id = [user_id for i in range(len(seqs_len))]
    skill_id_sequences, _ = sequence_data_augment(list(df['skill_id']),step_size=step_size)
    same_skill_total_num_sequences, _ = sequence_data_augment(list(df['same_skill_total_num']),step_size=step_size)
    same_skill_correct_num_sequences, _ = sequence_data_augment(list(df['same_skill_correct_num']),step_size=step_size)
    attempt_sequences, _ = sequence_data_augment(list(df['attempt_count']),step_size=step_size)
    correctness_sequences, _ = sequence_data_augment(list(df['correct']),step_size=step_size)

    df = pd.DataFrame(
        data={
            'user_id': user_id,
            'seq_len': seqs_len,
            'question_id_sequence': question_sequences,
            'skill_id_sequence': skill_id_sequences,
            'same_skill_total_num_sequence': same_skill_total_num_sequences,
            'same_skill_correct_num_sequence': same_skill_correct_num_sequences,
            'attempt_sequence': attempt_sequences,
            'correctness_sequence': correctness_sequences
        },
        columns=['user_id', 'seq_len','question_id_sequence', 'skill_id_sequence',
                 'same_skill_total_num_sequence', 'same_skill_correct_num_sequence','attempt_sequence', 'correctness_sequence']
    )
    return df


def train_val_test_split(user_df, train_test_ratio=0.7, train_val_ratio=0.8):
    train_len = int(len(user_df) * train_test_ratio)
    test_df = user_df.iloc[train_len:]
    train_df = user_df.iloc[0:train_len]

    train_len = int(train_len * train_val_ratio)
    val_df = train_df.iloc[train_len:]
    train_df = train_df.iloc[0:train_len]

    return train_df, val_df, test_df


def pre_process(root_dir, filename):
    # TODO 1: Read
    print('start to process the file : ',filename)
    df = pd.read_csv(
        os.path.join(root_dir, filename) + '.csv',
        dtype={'skill_name': 'str'},
        usecols=['user_id', 'problem_id', 'skill_id', 'correct', 'order_id','attempt_count','skill_name'],
    )
    print('successfully read the file :', filename)

    # TODO 2 : Dropna , DropDupliates, Sort, ResetIndex
    df = df[
        ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'attempt_count','skill_name']]
    df = df.dropna().drop_duplicates()
    df = df.sort_values(by=['user_id', 'order_id', 'problem_id'])
    df = df.reset_index()

    # TODO 3 : LabelTransform: user_id, skill_id, problem_id
    df = label_transform(df, 'user_id')
    df = label_transform(df, 'skill_id')
    df = label_transform(df, 'problem_id')
    print(df['skill_id'].value_counts())

    # TODO 4: CategoryTransform: attempt_count
    print('start to CategoryTransform {}'.format('attempt_count'))
    df['attempt_count'] = category_transform(df['attempt_count'])
    print('start to CategoryTransform {}'.format('attempt_count'))

    # TODO 5: Groupby: user_id
    grouped_df = df.groupby('user_id')
    users_list = list(grouped_df.groups.keys())

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for user_id in users_list:
        user_df = grouped_df.get_group(user_id)
        if (len(user_df)) < 50:
            continue
        print(len(user_df))

        # train val test split
        user_train_df, user_val_df, user_test_df = train_val_test_split(
            user_df,
            train_test_ratio=0.7,
            train_val_ratio=0.8
        )

        # 计算 skill正确率
        user_train_df = calculate_and_add_correctness_ratio(user_train_df)
        user_val_df = calculate_and_add_correctness_ratio(user_val_df)
        user_test_df = calculate_and_add_correctness_ratio(user_test_df)

        # 一个user_id 做成一个 list str 数据
        user_train_df = get_list_df(user_train_df,step_size=5)
        user_val_df = get_list_df(user_val_df,step_size=5)
        user_test_df = get_list_df(user_test_df,step_size=5)

        train_df = pd.concat([train_df, user_train_df])
        test_df = pd.concat([test_df, user_test_df])
        val_df = pd.concat([val_df, user_val_df])

    # TODO 6: save the csv
    print('start to save data in csv file')
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
    print('finish to save data in csv file\n\n')


def run():
    pre_process(
        root_dir='data/Assistment09',
        filename='skill_builder_data_corrected'
    )
    pre_process(
        root_dir='data/Assistment09',
        filename='skill_builder_data_corrected_small'
    )
    pre_process(
        root_dir='data/Assistment09',
        filename='skill_builder_data_corrected_big'
    )


run()
