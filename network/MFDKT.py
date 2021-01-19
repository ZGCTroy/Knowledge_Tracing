import torch
import torch.nn as nn

PAD_INDEX = 0

class MF(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, user_num, skill_num, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.model_name = 'MF'
        self.embedding_layer1 = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim,
                                             padding_idx=PAD_INDEX)
        self.embedding_layer2 = nn.Embedding(num_embeddings=skill_num, embedding_dim=embedding_dim,
                                             padding_idx=PAD_INDEX)

        self.linear1 = nn.Linear(in_features=2 * embedding_dim, out_features=hidden_dim)
        self.decoder = nn.Linear(in_features=output_dim, out_features=output_dim)

    def forward(self, user_id_sequence, skill_sequence):
        embedding_vector1 = self.embedding_layer1(user_id_sequence)
        embedding_vector2 = self.embedding_layer2(skill_sequence)

        embedding_vector = torch.cat([embedding_vector1, embedding_vector2], dim=2)

        hidden_vector = self.linear1(embedding_vector)

        return hidden_vector




# class MF(nn.Module):
#     def __init__(self, embedding_dim, user_num, skill_num):
#         super(MF, self).__init__()
#         self.P = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
#         self.Q = nn.Embedding(num_embeddings=skill_num, embedding_dim=embedding_dim)
#         self.P_bias = nn.Embedding(num_embeddings=user_num, embedding_dim=1)
#         self.Q_bias = nn.Embedding(num_embeddings=skill_num, embedding_dim=1)
#
#     def forward(self, user_id_sequence, skill_sequence):
#         P = self.P(user_id_sequence)
#         Q = self.Q(skill_sequence)
#         P_bias = self.P_bias(user_id_sequence)
#         Q_bias = self.Q_bias(skill_sequence)
#
#         output = torch.sum(P * Q,dim=1,keepdim=True) + P_bias + Q_bias
#
#         return output



class MFDKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, user_num,max_seq_len, skill_num):
        super().__init__()
        self.model_name = 'MFDKT'
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self._encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_INDEX)

        # self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.LSTMCell = nn.LSTMCell(input_size=embedding_dim,hidden_size=hidden_dim)
        self._decoder = nn.Linear(hidden_dim, output_dim)

        self.MF = MF(user_num=user_num, skill_num=skill_num,embedding_dim=skill_num, output_dim=hidden_dim)
        self.max_seq_len = max_seq_len

    def forward(self, input, target_id, user_id_sequence, skill_sequence, attempt_sequence):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        # TODO 1: get embedded input
        embedded_input = self._encoder(input)

        # TODO 2: get extended input
        # extended_input = self.MF.Q(attempt_sequence)
        extended_input = self.MF(user_id_sequence, skill_sequence)

        # TODO 3: Combine embedded input and extended input Before or During the LSTM
        # Combine Linear Add
        # embedded_input = torch.cat([embedded_input, extended_input],dim=2)
        # embedded_input = self.combine_linear(embedded_input)

        # DirectAdd
        # embedded_input += extended_input

        # Dot
        # embedded_input = embedded_input * extended_input

        ht = torch.zeros_like(embedded_input[:,0,:])
        ct = torch.zeros_like(embedded_input[:,0,:])
        for t in range(0,self.max_seq_len):
            # SigmoidAdd
            # ht = ht + torch.sigmoid(extended_input[:, t])

            # SigmoidDot
            ht = ht * torch.sigmoid(extended_input[:, t])

            # LSTM
            ht, ct = self.LSTMCell(embedded_input[:,t], (ht,ct))

        output = ht
        # output, _ = self._lstm(embedded_input)

        # TODO 4 : OUTPUT
        # hidden_dim --> question_embedding_dim (skill num)
        output = self._decoder(output)  # (batch_size, embedding_dim) ,embedding_dim = skill_num 表示该学生当前的对所有知识的掌握水平

        # target_id 即 skill id, 取出output 中 第skill_id 的值
        output = torch.gather(output, -1, target_id)

        output = torch.nn.Sigmoid()(output)

        return output
