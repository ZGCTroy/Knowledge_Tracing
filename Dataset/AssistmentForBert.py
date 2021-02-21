import pandas as pd
import torch
from torch.utils import data
import random
import numpy as np
from Dataset.data_helper import setup_seed, setup_pandas, strList_to_list


class AssistmentForBert(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv',
                 input_name=['skill_id_sequence'], vocab_size={}, label_name='last_correctness',
                 query_name='last_skill_id', trunk_size=15):
        self.path = path
        self.max_seq_len = 999
        self.min_seq_len = trunk_size + 1

        df = pd.read_csv(
            path,
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'problem_id_sequence', 'correctness_sequence',
                     'st+ct_sequence','power_sequence','skill_difficulty_sequence','problem_difficulty_sequence'],
        )

        df = df.dropna().drop_duplicates()
        df = df[df['seq_len'] >= self.min_seq_len]
        df = df.sort_values(by=['user_id'])
        df = df.reset_index()

        self.df = df
        self.data = {}
        self.truncated_data = {}
        self.pair_sequences = {}

        self.vocab_size = vocab_size

        self.input_name = input_name
        self.query_name = query_name
        self.label_name = label_name

        self.trunk_size = trunk_size

        self.truncated_data = {}
        self.preprocess()
        self.dataset_size = self.__len__()

    def __len__(self):
        return len(self.truncated_data['last_correctness'])

    def preprocess(self):
        self.data['user_id_sequence'] = list(self.df['user_id'])
        self.data['skill_id_sequence'] = [strList_to_list(strList, type='int') for strList in
                                          list(self.df['skill_id_sequence'])]
        self.data['problem_id_sequence'] = [strList_to_list(strList, type='int') for strList in
                                            list(self.df['problem_id_sequence'])]
        self.data['correctness_sequence'] = [strList_to_list(strList, type='int') for strList in
                                             list(self.df['correctness_sequence'])]
        self.data['st+ct_sequence'] = [strList_to_list(strList, type='int') for strList in
                                       list(self.df['st+ct_sequence'])]
        self.data['power_sequence'] = [strList_to_list(strList, type='int') for strList in
                                       list(self.df['power_sequence'])]
        self.data['skill_difficulty_sequence'] = [strList_to_list(strList, type='int') for strList in
                                       list(self.df['skill_difficulty_sequence'])]
        self.data['problem_difficulty_sequence'] = [strList_to_list(strList, type='int') for strList in
                                            list(self.df['problem_difficulty_sequence'])]

        self.truncated_data = {
            'skill_id_sequence': [],
            'problem_id_sequence': [],
            'correctness_sequence':[],
            'st+ct_sequence': [],
            'power_sequence':[],
            'skill_difficulty_sequence':[],
            'problem_difficulty_sequence': [],
            'last_skill_id': [],
            'last_problem_id': [],
            'last_correctness': [],
        }
        for i in range(len(self.data['skill_id_sequence'])):
            total_seq_len = len(self.data['skill_id_sequence'][i])
            for j in range(0, total_seq_len - self.trunk_size):
                self.truncated_data['skill_id_sequence'].append(self.data['skill_id_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['problem_id_sequence'].append(self.data['problem_id_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['correctness_sequence'].append(self.data['correctness_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['st+ct_sequence'].append(self.data['st+ct_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['power_sequence'].append(self.data['power_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['skill_difficulty_sequence'].append(self.data['skill_difficulty_sequence'][i][j:j + self.trunk_size])
                self.truncated_data['problem_difficulty_sequence'].append(
                    self.data['problem_difficulty_sequence'][i][j:j + self.trunk_size])

                self.truncated_data['last_skill_id'].append(self.data['skill_id_sequence'][i][j + self.trunk_size])
                self.truncated_data['last_correctness'].append(self.data['correctness_sequence'][i][j + self.trunk_size])
                self.truncated_data['last_problem_id'].append(self.data['problem_id_sequence'][i][j + self.trunk_size])

    def get_mask(self, seq_len):
        mask = [0 for i in range(seq_len)]
        for i in range(seq_len):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    mask[i] = 1
                # 10% randomly change token to random token
                elif prob < 0.9:
                    mask[i] = 2
                # 10% randomly change token to current token
                else:
                    mask[i] = 3

            else:
                mask[i]=0

        return mask

    def __getitem__(self, index):

        query = self.truncated_data[self.query_name][index]
        label = self.truncated_data[self.label_name][index]
        src_mask = [False] + [False] * self.trunk_size + [False]
        segment_info = [1] * (self.trunk_size + 2)

        data = {
            'query': torch.LongTensor([query]),
            'label': torch.FloatTensor([label]),
            'src_mask': torch.BoolTensor(src_mask),
            'segment_info': torch.LongTensor(segment_info)
        }

        mask = [0] + self.get_mask(seq_len=self.trunk_size) + [0]
        masked_data = {
            'mask': torch.BoolTensor(np.array(mask) > 0),
        }

        for name in self.input_name:
            sequence = self.truncated_data[name][index]

            pad_index = 0
            cls_index = 1 + self.vocab_size[name]
            eos_index = 2 + self.vocab_size[name]
            mask_index = 3 + self.vocab_size[name]

            sequence = [cls_index] + sequence + [eos_index]

            masked_sequence = [0 for i in sequence]
            for i in range(len(sequence)):
                if mask[i] == 0:
                    masked_sequence[i] = sequence[i]
                else:
                    # 80% randomly change token to mask token
                    if mask[i] == 1:
                        masked_sequence[i] = mask_index

                    # 10% randomly change token to random token
                    elif mask[i] == 2:
                        masked_sequence[i] = random.randrange(self.vocab_size[name]) + 1

                    # 10% randomly change token to current token
                    elif mask[i] == 3:
                        masked_sequence[i] = sequence[i]

            data[name] = torch.LongTensor(sequence)
            masked_data[name] = torch.LongTensor(masked_sequence)


        return data, masked_data




def print_train_data(i, trunk_size):
    # path = 'data/Assistment15/2015_100_skill_builders_main_problems.csv'
    # path = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed_train.csv'
    path = '../data/Assistment09/skill_builder_data_corrected_preprocessed_val.csv'
    # MAX_USER_NUM = 4151
    # SKILL_NUM = 110
    input_name = ['skill_id_sequence', 'correctness_sequence', 'st+ct_sequence', 'power_sequence',
                  'skill_difficulty_sequence','problem_difficulty_sequence']

    dataset = AssistmentForBert(
        path=path,
        vocab_size={
            'skill_id_sequence':110,
            'st+ct_sequence': 110 * 2,
            'correctness_sequence':2,
            'power_sequence':100,
            'skill_difficulty_sequence':10,
            'problem_difficulty_sequence':100
        },
        input_name=input_name,
        query_name='last_skill_id',
        label_name='last_correctness',
        trunk_size=trunk_size
    )
    data, masked_data = dataset.__getitem__(i)

    for name in input_name:
        print(name)
        print(data[name])
        print('mask')
        print(masked_data['mask'])
        print('masked',name)
        print(masked_data[name])
        print()

    print('src_mask\n', data['src_mask'])
    print('label\n', data['label'])
    print('query\n', data['query'])


def test():
    setup_seed(41)
    setup_pandas()
    for i in range(100):
        print_train_data(i, trunk_size=10)


# test()
