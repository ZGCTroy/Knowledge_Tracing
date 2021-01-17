import torch
import torch.nn as nn

PAD_INDEX = 0

#
# class MF(nn.Module):
#     """
#     LSTM based model
#     """
#
#     def __init__(self, user_num, skill_num, max_attempt_num, embedding_dim):
#         super().__init__()
#         self.model_name = 'MF'
#         self.embedding_layer1 = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim,
#                                              padding_idx=PAD_INDEX)
#         self.embedding_layer2 = nn.Embedding(num_embeddings=skill_num, embedding_dim=embedding_dim,
#                                              padding_idx=PAD_INDEX)
#         self.embedding_layer3 = nn.Embedding(num_embeddings=max_attempt_num, embedding_dim=embedding_dim,
#                                              padding_idx=PAD_INDEX)
#
#         self.linear1 = nn.Linear(in_features=3 * embedding_dim, out_features=embedding_dim)
#         self.decoder = nn.Linear(in_features=embedding_dim, out_features=skill_num + 1)
#
#     def forward(self, user_id_sequence, skill_sequence, attempt_sequence):
#         embedding_vector1 = self.embedding_layer1(user_id_sequence)
#         embedding_vector2 = self.embedding_layer1(skill_sequence)
#         embedding_vector3 = self.embedding_layer1(attempt_sequence)
#
#         embedding_vector = torch.cat([embedding_vector1, embedding_vector2, embedding_vector3], dim=2)
#
#         output = self.linear1(embedding_vector)
#         output = self.decoder(output)
#
#         return output
#
#     def get_embedding_vector(self, user_id_sequence, skill_sequence, attempt_sequence):
#         embedding_vector1 = self.embedding_layer1(user_id_sequence)
#         embedding_vector2 = self.embedding_layer1(skill_sequence)
#         embedding_vector3 = self.embedding_layer1(attempt_sequence)
#         embedding_vector = torch.cat([embedding_vector1, embedding_vector2, embedding_vector3], dim=2)
#         embedding_vector = self.linear1(embedding_vector)
#
#         return embedding_vector


class MF(nn.Module):
    def __init__(self, embedding_dim,skill_num, attempt_num):
        super(MF, self).__init__()
        self.P = nn.Embedding(num_embeddings=skill_num, embedding_dim=embedding_dim)
        self.Q = nn.Embedding(num_embeddings=attempt_num, embedding_dim=embedding_dim)
        self.P_bias = nn.Embedding(num_embeddings=skill_num, embedding_dim=1)
        self.Q_bias = nn.Embedding(num_embeddings=attempt_num, embedding_dim=1)

    def forward(self, skill_sequence, attempt_sequence):
        P = self.P(skill_sequence)
        Q = self.Q(attempt_sequence)
        P_bias = self.P_bias(skill_sequence)
        Q_bias = self.Q_bias(attempt_sequence)

        output = torch.sum(P * Q,dim=1,keepdim=True) + P_bias + Q_bias

        return output



class MFDKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, attempt_num):
        super().__init__()
        self.model_name = 'MFDKT'
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self._encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_INDEX)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._decoder = nn.Linear(hidden_dim, output_dim)

        # self.extended_model = extended_model
        # self.combine_linear = nn.Linear(embedding_dim * 2, embedding_dim)

        self.MF = MF(embedding_dim=embedding_dim,skill_num=output_dim, attempt_num=attempt_num)

    def forward(self, input, target_id, user_id_sequence, skill_sequence, attempt_sequence):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        embedded_input = self._encoder(input)

        # embedded_input += extended_input
        # embedded_input = torch.cat([embedded_input, extended_input],dim=2)
        # embedded_input = self.combine_linear(embedded_input)

        # extended_input = self.MF(skill_sequence,attempt_sequence)
        # embedded_input += extended_input

        output, _ = self._lstm(embedded_input)

        # hidden_dim --> question_embedding_dim (skill num)
        output = self._decoder(output[:, -1, :])  # (batch_size, embedding_dim) ,embedding_dim = skill_num 表示该学生当前的对所有知识的掌握水平

        # target_id 即 skill id, 取出output 中 第skill_id 的值
        output = torch.gather(output, -1, target_id)

        output = torch.nn.Sigmoid()(output)

        return output
