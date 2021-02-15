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
            usecols=['user_id', 'seq_len', 'skill_id_sequence', 'problem_id_sequence','correctness_sequence', 'skill_states',
                     'skill_states_mask', 'similar_user_id_in_train', 'ranked_similarity_in_train'],
        )

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
            l = [int(i) for i in l]
        return l

    def get_pre_part_of_sequence(self, seq):
        pre_part_of_sequence = seq[:self.max_seq_len]
        return pre_part_of_sequence

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

        # TODO 1: strList to List
        skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[:-1]
        next_skill_id_sequence = self.strList_to_list(data['skill_id_sequence'])[1:]
        problem_id_sequence = self.strList_to_list(data['problem_id_sequence'])[:-1]
        next_problem_id_sequence = self.strList_to_list(data['problem_id_sequence'])[1:]
        correctness_sequence = self.strList_to_list(data['correctness_sequence'])[:-1]
        next_correctness_sequence = self.strList_to_list(data['correctness_sequence'])[1:]
        skill_states = self.strList_to_list(data['skill_states'], type='float')
        skill_states_mask = self.strList_to_list(data['skill_states_mask'], type='bool')
        ranked_similarity_in_train = self.strList_to_list(data['ranked_similarity_in_train'],type='float')
        similar_user_id_in_train = self.strList_to_list(data['similar_user_id_in_train'],type='int')

        # TODO 2: get post part of sequence
        total_len = len(next_correctness_sequence)
        skill_id_sequence = self.get_pre_part_of_sequence(skill_id_sequence)
        next_skill_id_sequence = self.get_pre_part_of_sequence(next_skill_id_sequence)
        problem_id_sequence = self.get_pre_part_of_sequence(problem_id_sequence)
        next_problem_id_sequence = self.get_pre_part_of_sequence(next_problem_id_sequence)
        correctness_sequence = self.get_pre_part_of_sequence(correctness_sequence)
        next_correctness_sequence = self.get_pre_part_of_sequence(next_correctness_sequence)

        real_len = len(next_correctness_sequence)

        # TODO 3: add pre padding
        skill_id_sequence = self.add_pre_padding(skill_id_sequence, type='int')
        next_skill_id_sequence = self.add_pre_padding(next_skill_id_sequence, type='int')
        problem_id_sequence = self.add_pre_padding(problem_id_sequence, type='int')
        next_problem_id_sequence = self.add_pre_padding(next_problem_id_sequence, type='int')
        correctness_sequence = self.add_pre_padding(correctness_sequence, type='int')
        next_correctness_sequence = self.add_pre_padding(next_correctness_sequence, type='int')
        label_mask = (self.max_seq_len - real_len) * [0] + real_len * [1]

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'total_len': torch.LongTensor([total_len]),
            'skill_id_sequence': torch.LongTensor(skill_id_sequence),
            'next_skill_id_sequence': torch.LongTensor(next_skill_id_sequence),
            'problem_id_sequence': torch.LongTensor(problem_id_sequence),
            'next_problem_id_sequence': torch.LongTensor(next_problem_id_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'next_correctness_sequence': torch.FloatTensor(next_correctness_sequence),
            'skill_states': torch.FloatTensor(skill_states),
            'skill_states_mask': torch.BoolTensor(skill_states_mask),
            'similar_user_id_in_train':torch.LongTensor(similar_user_id_in_train),
            'ranked_similarity_in_train':torch.FloatTensor(ranked_similarity_in_train),
            'label_mask': torch.BoolTensor(label_mask)
        }


path = '../data/Assistment09/skill_builder_data_corrected_preprocessed_val.csv'
# path = 'data/Assistment15/2015_100_skill_builders_main_problems.csv'
# path = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed_train.csv'

def print_data(i, max_seq_len):
    dataset = Assistment(path=path, max_seq_len=max_seq_len)
    data = dataset.__getitem__(i)
    print(data['user_id'])
    print('total len = ', data['total_len'])
    print('real len = ', data['real_len'])
    print('skill_id_sequence\n',data['skill_id_sequence'])
    print('next_skill_id_sequence\n',data['next_skill_id_sequence'])
    print('problem_id_sequence\n', data['problem_id_sequence'])
    print('correctness_sequence\n',data['correctness_sequence'])
    print('next_correctness_sequence\n',data['next_correctness_sequence'])
    print('skill_states\n',data['skill_states'])
    print('similar_user_id\n',data['similar_user_id_in_train'])
    print('ranked_similarity\n',data['ranked_similarity_in_train'])
    print('skill_states_mask\n',data['skill_states_mask'])
    print('mask\n',data['label_mask'].shape,data['label_mask'])
    print()
    print()

# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
#
# mask = generate_square_subsequent_mask(4)
# print(mask.shape)
# print(mask)
# print_data(0,max_seq_len=20)
# print_data(0,max_seq_len=20)
# print_data(0,max_seq_len=40)

# path = '../data/Assistment09/skill_builder_data_corrected_preprocessed_train.csv'
# print_data(0,max_seq_len=10)
# print_data(73,max_seq_len=10)


