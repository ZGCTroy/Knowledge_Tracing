import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def calculate_and_add_correctness_ratio(df, mode='all'):
    df = pd.DataFrame(df)

    total_len = len(df)
    if mode == 'all':
        pos = max(0,int(total_len * 0.6)-1)
        grouped_df = df.iloc[:pos+1].groupby(by=['user_id', 'skill_id'])
    else:
        grouped_df = df.groupby(by=['user_id', 'skill_id'])

    same_skill_total_num = grouped_df['correct'].transform(len)
    same_skill_correct_num = grouped_df['correct'].transform(sum)
    df['same_skill_correctness_ratio'] = 0.5
    df['same_skill_correctness_ratio'].iloc[:pos+1] = same_skill_correct_num / same_skill_total_num
    print(df)
    df.fillna(0.5, inplace=True)
    print(df)

    return df

def train_val_test_split(user_df, train_test_ratio=0.7, train_val_ratio=0.8):
    train_len = int(len(user_df) * train_test_ratio)
    test_df = user_df.iloc[train_len:]
    train_df = user_df.iloc[0:train_len]

    train_len = int(train_len * train_val_ratio)
    val_df = train_df.iloc[train_len:]
    train_df = train_df.iloc[0:train_len]

    return train_df, val_df, test_df
def get_one_user_data(df, skill_num=110, mode='not all'):
    user_id = list(df['user_id'])[0]
    seq_len = len(list(df['user_id']))

    skill_id_sequence = str(list(df['skill_id']))
    correctness_sequence = str(list(df['correct']))

    SK = [0.5 for i in range(0,skill_num+1)]
    if mode=='all':
        pos = max(0, int(seq_len * 0.6) - 1)
    else:
        pos = seq_len - 1

    SK_total_num = [0 for i in range(0,skill_num+1)]
    SK_correct_num = [0 for i in range(0, skill_num + 1)]
    SK_mask = [False for i in range(0,skill_num+1)]

    for index in range(pos+1):

        skill_id = int(df['skill_id'].iloc[index])
        correct = df['correct'].iloc[index]
        SK_total_num[skill_id] += 1
        if correct == 1:
            SK_correct_num[skill_id] += 1
        SK[skill_id] = float(SK_correct_num[skill_id]) / float(SK_total_num[skill_id])
        SK_mask[skill_id] = True



def print_info(df):
    print()
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe(include='all'))
    print()
    print()