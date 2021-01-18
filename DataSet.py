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
            usecols=['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id', 'skill_name',
                     'attempt_count'],
        )

        df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'attempt_count', 'assistment_id',
                 'skill_name']]
        df = df.dropna().drop_duplicates()
        df = df.sort_values(by=['user_id', 'order_id', 'problem_id'])
        df = df.reset_index()

        df = df.groupby('user_id')

        self.user_list = list(df.groups.keys())
        self.df = df
        self.min_seq_len = 30

    def data_augment(self, user_id):
        user_df = self.df.get_group(user_id)
        if (len(user_df) < self.min_seq_len):
            return

        question_sequence = list(user_df['problem_id'])
        skill_sequence = list(user_df['skill_id'])
        correctness_sequence = list(user_df['correct'])
        attempt_sequence = list(user_df['attempt_count'])

        real_len = len(question_sequence) - 1

        if real_len >= self.max_sequence_len:
            query_question = question_sequence[self.max_sequence_len]
            query_skill = skill_sequence[self.max_sequence_len]
            query_correctness = correctness_sequence[self.max_sequence_len]

            # select pre max seq len
            question_sequence = question_sequence[:self.max_sequence_len]
            skill_sequence = skill_sequence[:self.max_sequence_len]
            correctness_sequence = correctness_sequence[:self.max_sequence_len]
            attempt_sequence = attempt_sequence[:self.max_sequence_len]

            # # select post max seq len
            # question_sequence = question_sequence[-self.max_sequence_len:]
            # skill_sequence = skill_sequence[-self.max_sequence_len:]
            # correctness_sequence = correctness_sequence[-self.max_sequence_len:]
        else:
            query_question = question_sequence[real_len]
            query_skill = skill_sequence[real_len]
            query_correctness = correctness_sequence[real_len]

            padding = (self.max_sequence_len - real_len) * [0]
            question_sequence = padding + question_sequence[:-1]
            skill_sequence = padding + skill_sequence[:-1]
            correctness_sequence = padding + correctness_sequence[:-1]
            attempt_sequence = padding + attempt_sequence[:-1]

    # def get_processed_data_list(self):
    #     for user_id in self.user_list:
    #         user_df = self.data_augment(user_id)
    #     self.final_df = self.final_df + user_df

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        user_id = self.user_list[index]
        user_df = self.df.get_group(user_id)

        question_sequence = list(user_df['problem_id'])
        skill_sequence = list(user_df['skill_id'])
        correctness_sequence = list(user_df['correct'])
        attempt_sequence = list(user_df['attempt_count'])

        real_len = len(question_sequence) - 1

        if real_len >= self.max_sequence_len:
            query_question = question_sequence[self.max_sequence_len]
            query_skill = skill_sequence[self.max_sequence_len]
            query_correctness = correctness_sequence[self.max_sequence_len]

            # select pre max seq len
            question_sequence = question_sequence[:self.max_sequence_len]
            skill_sequence = skill_sequence[:self.max_sequence_len]
            correctness_sequence = correctness_sequence[:self.max_sequence_len]
            attempt_sequence = attempt_sequence[:self.max_sequence_len]

            # # select post max seq len
            # question_sequence = question_sequence[-self.max_sequence_len:]
            # skill_sequence = skill_sequence[-self.max_sequence_len:]
            # correctness_sequence = correctness_sequence[-self.max_sequence_len:]
        else:
            query_question = question_sequence[real_len]
            query_skill = skill_sequence[real_len]
            query_correctness = correctness_sequence[real_len]

            padding = (self.max_sequence_len - real_len) * [0]
            question_sequence = padding + question_sequence[:-1]
            skill_sequence = padding + skill_sequence[:-1]
            correctness_sequence = padding + correctness_sequence[:-1]
            attempt_sequence = padding + attempt_sequence[:-1]

        return {
            'user_id': torch.LongTensor([user_id]),
            'real_len': torch.LongTensor([real_len]),
            'question_sequence': torch.LongTensor(question_sequence),
            'skill_sequence': torch.LongTensor(skill_sequence),
            'correctness_sequence': torch.LongTensor(correctness_sequence),
            'attempt_sequence': torch.LongTensor(attempt_sequence),
            'query_question': torch.LongTensor([query_question]),
            'query_skill': torch.LongTensor([int(query_skill)]),
            'query_correctness': torch.FloatTensor([query_correctness])
        }


# dataset = Assistment09()
#
# df = pd.read_csv(
#     'data/skill_builder_data_corrected_big.csv',
#     dtype={'skill_name': 'str'},
#     usecols=['user_id', 'assistment_id', 'problem_id', 'skill_id', 'correct', 'order_id', 'assistment_id',
#              'skill_name'],
# )
#
# # print(df['user_id'].drop_duplicates().count())
#
# print(df['correct'].value_counts())
