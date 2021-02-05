import pandas as pd
import torch
from torch.utils import data



class Assistment(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv', max_seq_len=200, min_seq_len=2):
        self.path = path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        df = pd.read_csv(
            path,
            dtype={'skill_name': 'str'},
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'correctness_sequence','skill_states','skill_states_mask'],
        )

        df = df[['user_id', 'seq_len', 'skill_id_sequence', 'correctness_sequence','skill_states','skill_states_mask']]

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
        elif type == 'float':
            l = [float(i) for i in l]
        elif type == 'bool':
            l = [bool(i) for i in l]
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
        skill_states = self.strList_to_list(data['skill_states'],type='float')
        skill_states_mask = self.strList_to_list(data['skill_states_mask'], type='bool')

        skill_id_sequence = self.get_post_part_of_sequence(skill_id_sequence)
        next_skill_id_sequence = self.get_post_part_of_sequence(next_skill_id_sequence)
        correctness_sequence = self.get_post_part_of_sequence(correctness_sequence)
        next_correctness_sequence = self.get_post_part_of_sequence(next_correctness_sequence)

        real_len = len(skill_id_sequence)
        int_padding = (self.max_seq_len - real_len) * [0]

        skill_id_sequence = int_padding + skill_id_sequence
        next_skill_id_sequence = int_padding + next_skill_id_sequence
        correctness_sequence = int_padding + correctness_sequence
        next_correctness_sequence = int_padding + next_correctness_sequence

        mask = (self.max_seq_len - real_len) * [False] + [True] * real_len

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'skill_id_sequence': torch.LongTensor(skill_id_sequence),
            'next_skill_id_sequence': torch.LongTensor(next_skill_id_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'next_correctness_sequence': torch.FloatTensor(next_correctness_sequence),
            'skill_states': torch.FloatTensor(skill_states),
            'skill_states_mask': torch.BoolTensor(skill_states_mask),
            'mask': torch.BoolTensor(mask)
        }

path = 'data/Assistment09/skill_builder_data_corrected_preprocessed_val.csv'
# path = 'data/Assistment15/2015_100_skill_builders_main_problems.csv'
# path = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed_train.csv'

#
# dataset = Assistment(path=path)
# for i in range(1):
#     data = dataset.__getitem__(i)
#     print(data['user_id'])
#     print(data['skill_id_sequence'])
#     print(data['next_skill_id_sequence'])
#     print(data['mask'])
#     print(data['correctness_sequence'])
#     print(data['next_correctness_sequence'])
#     print(data['same_skill_correctness_ratio_sequence'])



# df = pd.read_csv(
#     'data/skill_builder_data_corrected_big.csv',
#     dtype={'skill_name': 'str'},
#     usecols=['user_id', 'assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id',
#              'skill_name'],
# )

# print(df['user_id'].drop_duplicates().count())

# print(df['correct'].value_counts())
