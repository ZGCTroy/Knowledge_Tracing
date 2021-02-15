import pandas as pd
import torch
from torch.utils import data



class Assistment_new_version(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv', max_seq_len=400, min_seq_len=15,mode='train'):
        self.path = path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.mode = mode

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
        elif type=='float':
            l = [float(i) for i in l]
        elif type == 'bool':
            l = [bool(i) for i in l]
        return l

    def get_post_part_of_sequence(self, seq):
        post_part_of_sequence = seq[-self.max_seq_len:]
        return post_part_of_sequence

    def add_pre_padding(self, seq, type='int'):
        cur_len = len(seq)
        if cur_len < self.max_seq_len:
            if type == 'int':
                padding = (self.max_seq_len - cur_len) * [0]
            elif type=='float':
                padding = (self.max_seq_len - cur_len) * [0.]
            elif type=='bool':
                padding = (self.max_seq_len - cur_len) * [False]
            return padding + seq
        else:
            return seq

    def __getitem__(self, index):
        data = self.df.iloc[index]

        user_id = data['user_id']
        skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[:-1]
        next_skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[1:]
        correctness_sequence = self.strList_to_list(data['correctness_sequence'])[:-1]
        next_correctness_sequence = self.strList_to_list(data['correctness_sequence'])[1:]
        skill_states = self.strList_to_list(data['skill_states'],type='float')
        skill_states_mask = self.strList_to_list(data['skill_states_mask'],type='bool')

        total_len = len(skill_id_sequence)
        train_end_pos = max(0, int(0.6 * total_len) - 1)
        val_end_pos = max(0, int(0.8 * total_len) - 1)
        test_end_pos = max(0, total_len - 1)

        if self.mode == 'train':
            end_pos = train_end_pos
            real_len = train_end_pos + 1
            mask = [True] * (train_end_pos + 1)
        elif self.mode == 'val':
            end_pos = val_end_pos
            real_len = val_end_pos - train_end_pos
            mask = [False] * (train_end_pos + 1) + [True] * (val_end_pos - train_end_pos)
        elif self.mode == 'test':
            end_pos = test_end_pos
            real_len = test_end_pos - val_end_pos
            mask = [False] * (val_end_pos + 1) + [True] * (test_end_pos - val_end_pos)

        # TODO 2: get post part of sequence
        mask = self.get_post_part_of_sequence(mask)
        skill_id_sequence = self.get_post_part_of_sequence(skill_id_sequence[:end_pos+1])
        next_skill_id_sequence = self.get_post_part_of_sequence(next_skill_id_sequence[:end_pos+1])
        correctness_sequence = self.get_post_part_of_sequence(correctness_sequence[:end_pos+1])
        next_correctness_sequence = self.get_post_part_of_sequence(next_correctness_sequence[:end_pos+1])

        # TODO 3: add pre padding
        skill_id_sequence = self.add_pre_padding(skill_id_sequence, type='int')
        next_skill_id_sequence = self.add_pre_padding(next_skill_id_sequence, type='int')
        correctness_sequence = self.add_pre_padding(correctness_sequence, type='int')
        next_correctness_sequence = self.add_pre_padding(next_correctness_sequence, type='int')
        mask = self.add_pre_padding(mask, type='bool')

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'skill_id_sequence': torch.LongTensor(skill_id_sequence),
            'next_skill_id_sequence': torch.LongTensor(next_skill_id_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'next_correctness_sequence': torch.FloatTensor(next_correctness_sequence),
            'skill_states': torch.FloatTensor(skill_states),
            'skill_states_mask':torch.BoolTensor(skill_states_mask),
            'mask': torch.BoolTensor(mask)
        }

path = '../../data/Assistment09/skill_builder_data_corrected_preprocessed_val.csv'
# path = '../data/Assistment15/2015_100_skill_builders_main_problems_preprocessed.csv'
# path = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed_train.csv'


# dataset = Assistment_new_version(path=path, mode='val', max_seq_len=200,min_seq_len=15)
# for i in range(1):
#     data = dataset.__getitem__(i)
#     print(data['user_id'])
#     print(data['real_len'])
#     print(data['skill_id_sequence'])
#     print(data['next_skill_id_sequence'])
#     print(data['mask'])
#     print(data['correctness_sequence'])
#     print(data['next_correctness_sequence'])
#     print(data['skill_states'])



# df = pd.read_csv(
#     'data/skill_builder_data_corrected_big.csv',
#     dtype={'skill_name': 'str'},
#     usecols=['user_id', 'assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id',
#              'skill_name'],
# )

# print(df['user_id'].drop_duplicates().count())

# print(df['correct'].value_counts())
