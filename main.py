import os
import random

import numpy as np
import torch

from network.DKT import DKT
from network.MFDKT import MFDKT, MF
from solver.DKTSolver import DKTSolver
from solver.DKTMFSolver import DKTMFSolver


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    os.environ['PYTHONHASHSEED'] = str(seed)


def test_Baseline_DKT():
    QUESTION_MAX_NUM = 16891 + 1
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 111 + 1
    EMBEDDING_DIM = SKILL_NUM
    BATCH_SIZE = 64

    DKT_model = DKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=100,
        num_layers=1,
        output_dim=SKILL_NUM,
        dropout=0.2
    )

    DKT_solver = DKTSolver(
        log_name= 'SkillLevel_Baseline2',
        model=DKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed.csv',
        cuda='cuda:0',
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam(DKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    # DKT_solver.load_model(path='models_checkpoints/DKT.pt')

    DKT_solver.train(epochs=50)
    DKT_solver.test(DKT_solver.model)

    # DKT_solver.load_model(path='models_checkpoints/DKT_best_train_skillLevel_post.pt')


def test_MFDKT():
    USER_MAX_NUM = 4151
    QUESTION_MAX_NUM = 16891 + 1
    MAX_ATTEMPT_NUM = 100
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 111 + 1
    EMBEDDING_DIM = SKILL_NUM
    BATCH_SIZE = 64

    # MF_model = MF(
    #     user_num=USER_MAX_NUM,
    #     skill_num=SKILL_NUM,
    #     max_attempt_num=100,
    #     embedding_dim=EMBEDDING_DIM
    # )

    MFDKT_model = MFDKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=100,
        num_layers=1,
        output_dim=SKILL_NUM,
        dropout=0.2,
        attempt_num=50
    )

    # 'MF_skill+attempt_AddBeforeLSTM'
    MFDKT_solver = DKTMFSolver(
        log_name='DKT_BaseLine',
        model=MFDKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed.csv',
        cuda='cuda:0',
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam(MFDKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    # MFDKT_solver.load_model(path='models_checkpoints/DKT.pt')
    #
    MFDKT_solver.train(epochs=50)
    # MFDKT_solver.test(MFDKT_solver.model)

    # MFDKT_solver.load_model(path='models_checkpoints/DKT_best_train_skillLevel_post.pt')


if __name__ == '__main__':
    setup_seed(41)

    # test_Baseline_DKT()

    test_MFDKT()
