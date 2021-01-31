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
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv', max_seq_len=200, min_seq_len=2):
        self.path = path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        df = pd.read_csv(
            path,
            dtype={'skill_name': 'str'},
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'correctness_sequence',
                     'same_skill_total_num_sequence', 'same_skill_correct_num_sequence','skill_states'],
        )

        df = df[['user_id', 'seq_len', 'skill_id_sequence', 'correctness_sequence',
                 'same_skill_total_num_sequence', 'same_skill_correct_num_sequence','skill_states']]

        df = df.dropna().drop_duplicates()
        df = df[df['seq_len'] >= min_seq_len]
        df = df.sort_values(by=['user_id'])
        df = df.reset_index()

        self.df = df

    def __len__(self):
        return len(self.df)

    def strList_to_list(self, strList, type='int'):
        l = strList.strip('[],')
        l = l.split(',')
        if type == 'int':
            l = [int(i) for i in l]
        else:
            l = [float(i) for i in l]
        return l

    def get_post_part_of_sequence(self,seq):
        post_part_of_sequence = seq[-self.max_seq_len:]
        return post_part_of_sequence

    def __getitem__(self, index):
        data = self.df.iloc[index]
        user_id = data['user_id']

        skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[:-1]
        next_skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[1:]
        correctness_sequence = self.strList_to_list(data['correctness_sequence'])[:-1]
        next_correctness_sequence = self.strList_to_list(data['correctness_sequence'])[1:]
        same_skill_total_num_sequence = self.strList_to_list(data['same_skill_total_num_sequence'])[:-1]
        same_skill_correct_num_sequence = self.strList_to_list(data['same_skill_correct_num_sequence'])[:-1]
        same_skill_correctness_ratio_sequence = [float(a)/float(b) for a,b in zip(same_skill_correct_num_sequence,same_skill_total_num_sequence)]
        skill_states = self.strList_to_list(data['skill_states'],type='float')

        skill_id_sequence = self.get_post_part_of_sequence(skill_id_sequence)
        next_skill_id_sequence = self.get_post_part_of_sequence(next_skill_id_sequence)
        correctness_sequence = self.get_post_part_of_sequence(correctness_sequence)
        next_correctness_sequence = self.get_post_part_of_sequence(next_correctness_sequence)
        same_skill_correctness_ratio_sequence = self.get_post_part_of_sequence(same_skill_correctness_ratio_sequence)

        real_len = len(skill_id_sequence)
        float_padding = (self.max_seq_len - real_len) * [0.]
        int_padding = (self.max_seq_len - real_len) * [0]

        skill_id_sequence = int_padding + skill_id_sequence
        next_skill_id_sequence = int_padding + next_skill_id_sequence
        correctness_sequence = int_padding + correctness_sequence
        same_skill_correctness_ratio_sequence = float_padding + same_skill_correctness_ratio_sequence
        next_correctness_sequence = int_padding + next_correctness_sequence

        mask = (self.max_seq_len - real_len) * [False] + [True] * real_len

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'skill_id_sequence': torch.LongTensor(skill_id_sequence),
            'next_skill_id_sequence': torch.LongTensor(next_skill_id_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'next_correctness_sequence': torch.FloatTensor(next_correctness_sequence),
            'same_skill_correctness_ratio_sequence': torch.FloatTensor(same_skill_correctness_ratio_sequence),
            'skill_states': torch.FloatTensor(skill_states),
            'mask': torch.BoolTensor(mask)
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
