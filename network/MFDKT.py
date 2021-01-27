import torch
import torch.nn as nn

PAD_INDEX = 0
CUDA_LAUNCH_BLOCKING=1

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
            embedding_dim=1,
            padding_idx=0
        )
        self.Q_bias = nn.Embedding(
            num_embeddings=skill_num + 1,
            embedding_dim=1,
            padding_idx=0
        )

    def forward(self, user_id_sequence, skill_id_sequence):
        P = self.P(user_id_sequence)
        Q = self.Q(skill_id_sequence)
        P_bias = self.P_bias(user_id_sequence)
        Q_bias = self.Q_bias(skill_id_sequence)

        output = torch.sum(P * Q, dim=2, keepdim=True) + P_bias + Q_bias
        output = torch.sigmoid(output)

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

        self.encoder = nn.Embedding(
            num_embeddings=2 * skill_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=PAD_INDEX,
            _weight=torch.zeros(2 * skill_num + 1, embedding_dim)
        )
        self.dropLayer = nn.Dropout(dropout)

        self.LSTMCell = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_dim)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.MF = MF(user_num=user_num, skill_num=skill_num, embedding_dim=hidden_dim)
        self.max_seq_len = max_seq_len
        self.MLP = nn.Sequential(
            nn.Linear(output_dim+skill_num+1, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, main_input, target_id, user_id_sequence, skill_id_sequence):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        # TODO 1: get embedded input
        embedded_input = self.encoder(main_input)
        embedded_input = self.dropLayer(embedded_input)

        # TODO 2: get extended input
        # extended_input = self.MF(user_id_sequence, skill_id_sequence)
        # et = extended_input[:, -1]
        extended_input = self.MF.get_SK(user_id_sequence[:,0])

        ht = torch.zeros_like(embedded_input[:, 0, :])
        ct = torch.zeros_like(embedded_input[:, 0, :])
        for t in range(0, self.max_seq_len):
            xt = embedded_input[:, t]
            # et = extended_input[:, t]

            ht, ct = self.LSTMCell(xt, (ht, ct))

        # CombineAdd

        # output = torch.cat([ht,et],dim=1)

        # TODO 4 : OUTPUT
        # hidden_dim --> question_embedding_dim (skill num)
        output = self.decoder(ht)  # (batch_size, embedding_dim) ,embedding_dim = skill_num 表示该学生当前的对所有知识的掌握水平
        output = torch.cat([output,extended_input],dim=1)
        output = self.MLP(output)

        # target_id 即 skill id, 取出output 中 第skill_id 的值
        output = torch.gather(output, -1, target_id)

        output = torch.nn.Sigmoid()(output)

        return output
