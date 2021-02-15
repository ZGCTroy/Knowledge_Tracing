import random

import pandas as pd
import torch
from torch.utils import data

from .data_helper import setup_seed, setup_pandas


class AssistmentForBertPrtrain(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv',
                 input_name='skill_id_seuqnece', label_name='skill_id_sequence', input_vocab_size=110,
                 label_vocab_size=110, trunk_size=15):
        self.path = path
        self.min_seq_len = 2 * trunk_size

        df = pd.read_csv(
            path,
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'problem_id_sequence', 'correctness_sequence'],
        )
        df = df.dropna().drop_duplicates()
        df = df[df['seq_len'] >= min_seq_len]
        df = df.sort_values(by=['user_id'])
        df = df.reset_index()

        self.df = df
        self.data = {}
        self.truncated_data = {}
        self.pair_sequences = {}

        self.pad_index = 0
        self.sos_index = input_vocab_size + 1
        self.eos_index = input_vocab_size + 2
        self.mask_index = input_vocab_size + 3

        self.input_name = input_name
        self.label_name = label_name
        self.input_vocab_size = input_vocab_size
        self.label_vocab_size = label_vocab_size
        self.trunk_size = trunk_size

        self.preprocess()
        self.dataset_size = self.__len__()

    def __len__(self):
        return len(self.pair_sequences[self.input_name])

    def strList_to_list(self, strList, type='int'):
        l = strList.strip('[],')
        l = l.split(',')
        if type == 'int':
            l = [int(i) for i in l]
        elif type == 'float':
            l = [float(i) for i in l]
        elif type == 'bool':
            l = [int(i) for i in l]
        return l

    def preprocess(self):
        self.data['user_id_sequence'] = list(self.df['user_id'])
        self.data['skill_id_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                           list(self.df['skill_id_sequence'])]
        self.data['problem_id_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                             list(self.df['problem_id_sequence'])]
        self.data['correctness_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                              list(self.df['correctness_sequence'])]

        trunk_len = 30
        self.truncated_data['skill_id_sequences'] = self.get_truncated_sequences(self.data['skill_id_sequences'],
                                                                                 self.trunk_size)
        self.truncated_data['problem_id_sequences'] = self.get_truncated_sequences(self.data['problem_id_sequences'],
                                                                                   self.trunk_size)
        self.truncated_data['correctness_sequences'] = self.get_truncated_sequences(self.data['correctness_sequences'],
                                                                                    self.trunk_size)

        self.pair_sequences['skill_id_sequences'] = self.get_pair_sequences(self.truncated_data['skill_id_sequences'])
        self.pair_sequences['problem_id_sequences'] = self.get_pair_sequences(
            self.truncated_data['problem_id_sequences'])
        self.pair_sequences['correctness_sequences'] = self.get_pair_sequences(
            self.truncated_data['correctness_sequences'])

    def get_pair_sequences(self, all_truncated_sequences):
        pair_sequences = []
        for truncated_sequences in all_truncated_sequences:
            for i in range(len(truncated_sequences) - 1):
                pair_sequences.append([truncated_sequences[i], truncated_sequences[i + 1]])
        return pair_sequences

    def get_truncated_sequences(self, sequences, trunk_size=30):
        truncated_sequences = []
        for sequence in sequences:
            total_seq_len = len(sequence)
            truncated_sequence = []
            for i in range(0, total_seq_len, trunk_size):
                if i + trunk_size <= total_seq_len:
                    truncated_sequence.append(sequence[i:i + trunk_size])
            if len(truncated_sequence) >= 2:
                truncated_sequences.append(truncated_sequence)
        return truncated_sequences

    def get_masked_sequence(self, input, label):
        masked_input = [i for i in range(len(input))]
        masked_label = [i for i in range(len(label))]
        valid_label_mask = [False for i in range(len(label))]

        for i in range(len(input)):
            prob = random.random()
            if prob < 0.15:
                valid_label_mask[i] = True
                masked_label[i] = label[i]
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_input[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_input[i] = random.randrange(self.input_vocab_size) + 1

                # 10% randomly change token to current token
                else:
                    masked_input[i] = input[i]

            else:
                valid_label_mask[i] = False
                masked_input[i] = input[i]
                masked_label[i] = 0

        return masked_input, masked_label, valid_label_mask

    def __getitem__(self, index):
        input = self.pair_sequences[self.input_name][index]
        label = self.pair_sequences[self.label_name][index]
        nsp_label = 1

        if random.random() > 0.5:
            nsp_label = 0
            random_index = random.randrange(0, self.dataset_size)
            input[1] = self.pair_sequences[self.input_name][random_index][1]
            label[1] = self.pair_sequences[self.label_name][random_index][1]

        masked_input1, masked_label1, valid_label_mask1 = self.get_masked_sequence(input[0], label[0])
        masked_input2, masked_label2, valid_label_mask2 = self.get_masked_sequence(input[1], label[1])

        valid_label_mask = [False] + valid_label_mask1 + [False] + valid_label_mask2 + [False]
        segment_info = [1] + [1] * len(masked_input1) + [1] + [2] * len(masked_input2) + [2]
        masked_input = [self.sos_index] + masked_input1 + [self.eos_index] + masked_input2 + [self.eos_index]
        masked_label = [self.sos_index] + masked_label1 + [self.eos_index] + masked_label2 + [self.eos_index]

        # padding
        src_mask = [False] * len(masked_input) + [True] * (self.max_seq_len - len(masked_input))
        valid_label_mask = valid_label_mask + [False] * (self.max_seq_len - len(valid_label_mask))
        masked_input = masked_input + [self.pad_index] * (self.max_seq_len - len(masked_input))
        masked_label = masked_label + [self.pad_index] * (self.max_seq_len - len(masked_label))
        segment_info = segment_info + [self.pad_index] * (self.max_seq_len - len(segment_info))

        return {
            'nsp_label': torch.LongTensor([nsp_label]),
            'masked_input': torch.LongTensor(masked_input),
            'src_mask': torch.BoolTensor(src_mask),
            'masked_label': torch.LongTensor(masked_label),
            'valid_label_mask': torch.BoolTensor(valid_label_mask),
            'segment_info': torch.LongTensor(segment_info)
        }


class AssistmentForBert(data.Dataset):
    def __init__(self, path='data/skill_builder_data_corrected_preprocessed.csv',
                 input_name='skill_id_sequences', label_name='skill_id_sequences', query_name='skill_id_querys',input_vocab_size=110,
                 label_vocab_size=110, trunk_size=15):
        self.path = path
        self.max_seq_len = 999
        self.min_seq_len = trunk_size + 1

        df = pd.read_csv(
            path,
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'problem_id_sequence', 'correctness_sequence', ],
        )

        df = df.dropna().drop_duplicates()
        df = df[df['seq_len'] >= self.min_seq_len]
        df = df.sort_values(by=['user_id'])
        df = df.reset_index()

        self.df = df
        self.data = {}
        self.truncated_data = {}
        self.pair_sequences = {}

        self.pad_index = 0
        self.sos_index = input_vocab_size + 1
        self.eos_index = input_vocab_size + 2
        self.mask_index = input_vocab_size + 3

        self.input_name = input_name
        self.query_name = query_name
        self.label_name = label_name
        self.input_vocab_size = input_vocab_size
        self.label_vocab_size = label_vocab_size
        self.trunk_size = trunk_size

        self.truncated_data = {}
        self.preprocess()
        self.dataset_size = self.__len__()

    def __len__(self):
        return len(self.truncated_data['correctness'])

    def strList_to_list(self, strList, type='int'):
        l = strList.strip('[],')
        l = l.split(',')
        if type == 'int':
            l = [int(i) for i in l]
        elif type == 'float':
            l = [float(i) for i in l]
        elif type == 'bool':
            l = [int(i) for i in l]
        return l

    def preprocess(self):
        self.data['user_id_sequence'] = list(self.df['user_id'])
        self.data['skill_id_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                           list(self.df['skill_id_sequence'])]
        self.data['problem_id_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                             list(self.df['problem_id_sequence'])]
        self.data['correctness_sequences'] = [self.strList_to_list(strList, type='int') for strList in
                                              list(self.df['correctness_sequence'])]
        self.truncated_data={
            'skill_id_sequences':[],
            'skill_id_querys':[],
            'problem_id_sequences':[],
            'problem_id_querys':[],
            'correctness':[]
        }
        for i in range(len(self.data['skill_id_sequences'])):
            total_seq_len = len(self.data['skill_id_sequences'][i])
            for j in range(0, total_seq_len - self.trunk_size - 1):
                self.truncated_data['skill_id_sequences'].append(self.data['skill_id_sequences'][i][j:j + self.trunk_size])
                self.truncated_data['skill_id_querys'].append(self.data['skill_id_sequences'][i][j + self.trunk_size])
                self.truncated_data['problem_id_sequences'].append(self.data['problem_id_sequences'][i][j:j + self.trunk_size])
                self.truncated_data['problem_id_querys'].append(self.data['problem_id_sequences'][i][j + self.trunk_size])
                self.truncated_data['correctness'].append(self.data['correctness_sequences'][i][j + self.trunk_size])

    def __getitem__(self, index):

        input1 = self.truncated_data[self.input_name][index]

        segment_info = [1] + [1] * self.trunk_size + [1] + [2] * self.trunk_size + [2]
        input = [self.sos_index] + input1 + [self.eos_index] + [self.pad_index] * self.trunk_size + [self.eos_index]
        query = self.truncated_data[self.query_name][index]
        label = self.truncated_data[self.label_name][index]
        src_mask = [False] + [False] * self.trunk_size + [False] + [True] * self.trunk_size + [False]

        return {
            'input': torch.LongTensor(input),
            'query': torch.LongTensor([query]),
            'label': torch.FloatTensor([label]),
            'src_mask': torch.BoolTensor(src_mask),
            'segment_info': torch.LongTensor(segment_info)
        }


# path = '../data/Assistment09/skill_builder_data_corrected_preprocessed_val.csv'
# MAX_USER_NUM = 4151
# SKILL_NUM = 110

# path = 'data/Assistment15/2015_100_skill_builders_main_problems.csv'
# path = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed_train.csv'




def print_pretrain_data(i, max_seq_len):
    setup_seed(41)
    setup_pandas()
    dataset = AssistmentForBertPrtrain(
        path=path,
        input_vocab_size=110,
        label_vocab_size=110,
        input_name='skill_id_sequences',
        label_name='skill_id_sequences',
        trunk_size=15
    )
    data = dataset.__getitem__(i)
    print('nsp_label\n', data['nsp_label'])
    print('src_mask\n', data['src_mask'])
    print('masked_input\n', data['masked_input'])
    print('masked_label\n', data['masked_label'])
    print('valid_label_mask\n', data['valid_label_mask'])

def print_train_data(i, trunk_size):
    dataset = AssistmentForBert(
        path=path,
        input_vocab_size=110,
        label_vocab_size=110,
        input_name='skill_id_sequences',
        label_name='correctness',
        trunk_size=trunk_size
    )
    data = dataset.__getitem__(i)
    print('input\n', data['input'])
    print('src_mask\n', data['src_mask'])
    print('label\n', data['label'])
    print('query\n', data['query'])
    print('segment_info\n', data['segment_info'])

# for i in range(100):
#     print_train_data(i, max_seq_len=33)

