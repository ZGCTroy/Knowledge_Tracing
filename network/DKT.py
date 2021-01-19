import torch
import torch.nn as nn

PAD_INDEX = 0


class DKT(nn.Module):
    """
    LSTM based model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.model_name = 'DKT'
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_INDEX)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, target_id):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, max_sequence_len)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1), representing the probability of correctly answering the qt
        """

        embedded_input = self._encoder(input)

        output, _ = self._lstm(embedded_input)

        # hidden_dim --> question_embedding_dim (skill num)
        output = self._decoder(output[:, -1, :])  # (batch_size, embedding_dim) ,embedding_dim = skill_num 表示该学生当前的对所有知识的掌握水平

        # 1 : target_id 即 skill id, 取出output 中 第skill_id 的值
        output = torch.gather(output, -1, target_id)

        # 2 : weighted
        # query_embedded = self._encoder(target_id).view(batch_size,-1)
        # output = output * query_embedded
        # output = torch.sum(output,dim=1,keepdim=True)

        output = torch.nn.Sigmoid()(output)

        return output
