import torch.nn as nn


class BertLanguageModel(nn.Module):
    def __init__(self, bert_encoder, nsp_decoder, mlm_decoder, task_decoder=None):
        super(BertLanguageModel, self).__init__()
        self.model_name = 'Bert'
        self.bert_encoder = bert_encoder

        self.nsp_decoder = nsp_decoder

        self.mlm_decoder = mlm_decoder

        self.task_decoder = task_decoder

    def forward(self, src, segment_info, src_mask):
        embedding_output = self.bert_encoder(src=src, segment_info=segment_info, src_mask=src_mask)

        nsp_output = self.nsp_decoder(embedding_output[0])

        mlm_output = self.mlm_decoder(embedding_output)

        task_output = self.task_decoder(embedding_output[0])

        return embedding_output.permute(1,0,2), nsp_output, mlm_output.permute(1,0,2), task_output

class MlmDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim=2,dropout=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, input):
        return self.decoder(input)


class NspDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim,dropout=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, input):
        return self.decoder(input)

class TaskDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, dropout=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.decoder(input)
