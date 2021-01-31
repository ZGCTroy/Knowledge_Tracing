import pandas as pd
import torch
from torch.utils import data

# 显示所有列
pd.set_option('display.max_columns', 200)

# 显示所有行
pd.set_option('display.max_rows', None)

# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)
pd.set_option('expand_frame_repr', False)


class Assistment09(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_small.csv', max_seq_len=20, min_seq_len=5):
        self.path = path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        df = pd.read_csv(
            path,
            dtype={'skill_name': 'str'},
            usecols=['user_id', 'seq_len', 'question_id_sequence', 'skill_id_sequence', 'correctness_sequence',
                     'same_skill_total_num_sequence', 'same_skill_correct_num_sequence'],
        )

        df = df[['user_id', 'seq_len', 'question_id_sequence', 'skill_id_sequence', 'correctness_sequence',
                 'same_skill_total_num_sequence', 'same_skill_correct_num_sequence']]

        df = df.dropna().drop_duplicates()
        df = df[(df['seq_len'] >= min_seq_len) & (df['seq_len'] <= (max_seq_len + 1))]
        df = df.sort_values(by=['user_id'])
        df = df.reset_index()

        self.df = df

    def __len__(self):
        return len(self.df)

    def strList_to_list(self, strList):
        l = strList.strip('[],')
        l = l.split(',')
        l = [int(i) for i in l]
        return l

    def __getitem__(self, index):
        data = self.df.iloc[index]
        user_id = data['user_id']
        real_len = data['seq_len'] - 1

        question_id_sequence = self.strList_to_list(data['question_id_sequence'])
        skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])
        correctness_sequence = self.strList_to_list(data['correctness_sequence'])
        same_skill_total_num_sequence = self.strList_to_list(data['same_skill_total_num_sequence'])
        same_skill_correct_num_sequence = self.strList_to_list(data['same_skill_correct_num_sequence'])
        same_skill_correctness_ratio_sequence = [float(a)/float(b) for a,b in zip(same_skill_correct_num_sequence,same_skill_total_num_sequence)]


        query_question_id = question_id_sequence[-1]
        query_skill_id = skill_id_sequence[-1]
        query_correctness = correctness_sequence[-1]
        padding = (self.max_seq_len - real_len) * [0]
        float_padding = (self.max_seq_len - real_len) * [0.]
        question_id_sequence = padding + question_id_sequence[:-1]
        skill_id_sequence = padding + skill_id_sequence[:-1]
        correctness_sequence = padding + correctness_sequence[:-1]
        same_skill_correctness_ratio_sequence = float_padding + same_skill_correctness_ratio_sequence[:-1]

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'question_id_sequence': torch.LongTensor(question_id_sequence),
            'skill_id_sequence': torch.LongTensor(skill_id_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'same_skill_correctness_ratio_sequence': torch.FloatTensor(same_skill_correctness_ratio_sequence),
            'query_question_id': torch.LongTensor([query_question_id]),
            'query_skill_id': torch.LongTensor([query_skill_id]),
            'query_correctness': torch.FloatTensor([query_correctness])
        }


# dataset = Assistment09(path='data/skill_builder_data_corrected_preprocessed_train.csv')
# for i in range(1000,1010):
#     data = dataset.__getitem__(i)
#     print(data['user_id'])
#     print(data['skill_id_sequence'])
#     print(data['same_skill_correctness_ratio_sequence'])
#     print()


# df = pd.read_csv(
#     'data/skill_builder_data_corrected_big.csv',
#     dtype={'skill_name': 'str'},
#     usecols=['user_id', 'assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id',
#              'skill_name'],
# )

# print(df['user_id'].drop_duplicates().count())

# print(df['correct'].value_counts())
