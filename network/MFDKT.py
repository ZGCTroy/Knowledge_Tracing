import torch
import torch.nn as nn

PAD_INDEX = 0
CUDA_LAUNCH_BLOCKING=1

# class MF(nn.Module):
#     def __init__(self, user_num, skill_num,embedding_dim,):
#         super(MF, self).__init__()
#         self.P = nn.Embedding(
#             num_embeddings=user_num + 1,
#             embedding_dim=embedding_dim,
#             padding_idx=0
#         )
#         self.Q = nn.Embedding(
#             num_embeddings=skill_num + 1,
#             embedding_dim=embedding_dim,
#             padding_idx=0
#         )
#         self.P_bias = nn.Embedding(
#             num_embeddings=user_num + 1,
#             embedding_dim=1,
#             padding_idx=0
#         )
#         self.Q_bias = nn.Embedding(
#             num_embeddings=skill_num + 1,
#             embedding_dim=1,
#             padding_idx=0
#         )
#
#     def forward(self, user_id_sequence, skill_id_sequence):
#         P = self.P(user_id_sequence)
#         Q = self.Q(skill_id_sequence)
#         P_bias = self.P_bias(user_id_sequence)
#         Q_bias = self.Q_bias(skill_id_sequence)
#
#         output = torch.sum(P * Q, dim=2, keepdim=True) + P_bias + Q_bias
#         output = torch.sigmoid(output)
#
#         return output
#
#     def get_SK(self, user_id):
#         P = self.P(user_id)
#         Q = self.Q.weight
#         P_bias = self.P_bias(user_id)
#         Q_bias = self.Q_bias.weight
#
#         SK = torch.matmul(P, Q.transpose(0, 1)) + P_bias + Q_bias.transpose(0, 1)
#         SK = torch.sigmoid(SK)
#
#         return SK

class MF(nn.Module):
    def __init__(self, user_num, skill_num,embedding_dim,):
        super(MF, self).__init__()
        self.P = nn.Embedding(
            num_embeddings=user_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.Q = nn.Embedding(
            num_embeddings=skill_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.P_bias = nn.Embedding(
            num_embeddings=user_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.Q_bias = nn.Embedding(
            num_embeddings=skill_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, user_id_sequence, skill_id_sequence):
        P = self.P(user_id_sequence)
        Q = self.Q(skill_id_sequence)
        P_bias = self.P_bias(user_id_sequence)
        Q_bias = self.Q_bias(skill_id_sequence)

        output = P * Q  + P_bias + Q_bias
        output = self.decoder(output)
        output = torch.sigmoid(output)

        return output

    def get_embedding_vector(self, user_id_sequence, skill_id_sequence):
        P = self.P(user_id_sequence)
        Q = self.Q(skill_id_sequence)
        P_bias = self.P_bias(user_id_sequence)
        Q_bias = self.Q_bias(skill_id_sequence)

        output = P * Q + P_bias + Q_bias

        return output

    def get_SK(self, user_id):
        P = self.P(user_id)
        Q = self.Q.weight
        P_bias = self.P_bias(user_id)
        Q_bias = self.Q_bias.weight

        SK = torch.matmul(P, Q.transpose(0, 1)) + P_bias + Q_bias.transpose(0, 1)
        SK = torch.sigmoid(SK)

        return SK

class MFDKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self,user_num, skill_num, embedding_dim, hidden_dim, output_dim, num_layers, dropout,  max_seq_len
                 ):
        super().__init__()
        self.model_name = 'MFDKT'
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.encoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=2 * skill_num + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            ),
            nn.Dropout(dropout)
        )

        self.LSTM = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.MF = MF(
            user_num=user_num,
            skill_num=skill_num,
            embedding_dim=hidden_dim
        )

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim+embedding_dim, output_dim),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, main_input, target_id, user_id_sequence, skill_id_sequence):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        # TODO 1: get embedded input
        embedded_input = self.encoder(main_input)

        # TODO 2: get extended input
        # extended_input = self.MF(user_id_sequence, skill_id_sequence)
        extended_input = self.MF.get_embedding_vector(user_id_sequence, skill_id_sequence)
        # extended_input = self.MF.get_SK(user_id_sequence[:,0])
        # et = extended_input[:, -1]

        batch_size = main_input.shape[0]
        hidden = self.init_hidden(batch_size)
        LSTM_input = embedded_input
        # LSTM_input = torch.cat([embedded_input, extended_input], dim=2)
        h, hT = self.LSTM(LSTM_input, (hidden[0].detach(), hidden[1].detach()))

        # TODO 4 : OUTPUT
        # decoder_input = h
        decoder_input = torch.cat([h, extended_input],dim=2)
        output = self.decoder(decoder_input)

        # TODO 5: target_id 即 skill id, 取出output 中 第skill_id 的值
        output = torch.gather(output, dim=2, index = target_id.unsqueeze(dim=2)).squeeze(-1)

        return output
