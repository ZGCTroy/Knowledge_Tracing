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
    def __init__(self, path='data/skill_builder_data_corrected_small.csv', max_sequence_len=20):

        self.path = path
        self.max_sequence_len = max_sequence_len

        df = pd.read_csv(
            path,
            dtype={'skill_name': 'str'},
            usecols=['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id', 'skill_name'],
        )

        df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id', 'skill_name']]
        df = df.dropna().drop_duplicates()
        df = df.sort_values(by=['user_id', 'order_id', 'problem_id'])
        df = df.reset_index()

        df = df.groupby('user_id')

        self.user_list = list(df.groups.keys())
        self.df = df

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):

        user_id = self.user_list[index]
        user_df = self.df.get_group(user_id)

        question_sequence = list(user_df['problem_id'])
        skill_sequence = list(user_df['skill_id'])
        correctness_sequence = list(user_df['correct'])

        real_len = len(question_sequence) - 1

        if real_len >= self.max_sequence_len:
            query_question = question_sequence[self.max_sequence_len]
            query_skill = skill_sequence[self.max_sequence_len]
            query_correctness = correctness_sequence[self.max_sequence_len]

            question_sequence = question_sequence[0:self.max_sequence_len ]
            print(' ', self.max_sequence_len -1, len(question_sequence))
            skill_sequence = skill_sequence[:self.max_sequence_len - 1]
            correctness_sequence = correctness_sequence[:self.max_sequence_len - 1]
        else:
            query_question = question_sequence[real_len-1]
            query_skill = skill_sequence[real_len-1]
            query_correctness = correctness_sequence[real_len-1]

            question_sequence = question_sequence[:-1] + (self.max_sequence_len - real_len) * [0]
            skill_sequence = skill_sequence[:-1] + (self.max_sequence_len - real_len) * [0]
            correctness_sequence = correctness_sequence[:-1] + (self.max_sequence_len - real_len) * [0]

        print(real_len, len(question_sequence))
        return {
            'user_id': torch.Tensor([user_id]).int(),
            'real_len': torch.Tensor(real_len),
            'question_sequence': torch.LongTensor(question_sequence),
            'skill_sequence': torch.LongTensor(skill_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'query_question': torch.LongTensor(query_question),
            'query_skill': torch.LongTensor(int(query_skill)),
            'query_correctness': torch.LongTensor(query_correctness)
        }

# dataset = Assistment09()
# print(dataset.__getitem__(0))

# df = pd.read_csv(
#             'data/skill_builder_data_corrected_small.csv',
#             dtype={'skill_name': 'str'},
#             usecols=['user_id','assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id', 'skill_name'],
#         )
#
# print(df['skill_id'].drop_duplicates().count())
