import os
import random

import numpy as np
import torch

from network.DKT import DKT
from solver.DKTSolver import DKTSolver


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    os.environ['PYTHONHASHSEED'] = str(seed)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    setup_seed(41)

    QUESTION_MAX_NUM = 16891 + 1
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 111 + 1
    BATCH_SIZE = 64

    DKT_model = DKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=100,
        hidden_dim=100,
        num_layers=1,
        output_dim = SKILL_NUM,
        dropout=0.2
    )

    DKT_solver = DKTSolver(
        model=DKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed.csv',
        cuda = 'cuda:0',
        batch_size= BATCH_SIZE,
        optimizer=torch.optim.Adam(DKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    # DKT_solver.load_model(path='models_checkpoints/DKT_best_train_skillLevel_pre200.pt')

    DKT_solver.train(epochs=50)
    DKT_solver.test(DKT_solver.model)

    # DKT_solver.load_model(path='models_checkpoints/DKT_best_train_skillLevel_post.pt')
