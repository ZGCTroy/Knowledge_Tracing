import os
import random

import numpy as np
import torch

from network.DKT import DKT
from network.MFDKT import MFDKT
from network.PreMFDKT import PreMFDKT
from solver.DKTMFSolver import DKTMFSolver
from solver.DKTSolver import DKTSolver
from solver.PreDKTMFSolver import PreMFDKTSolver



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
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BATCH_SIZE = 64

    DKT_model = DKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM,
        dropout=0.2
    )

    DKT_solver = DKTSolver(
        log_name='SkillLevel/Baseline',
        model=DKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed',
        cuda='cuda:0',
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.AdamW(DKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    DKT_solver.train(epochs=50)

    DKT_solver.load_model(path='models_checkpoints/DKT/SkillLevel/Baseline.pt')
    DKT_solver.test(DKT_solver.model, mode='val')
    DKT_solver.test(DKT_solver.model, mode='test')


def test_MFDKT():
    MAX_USER_NUM = 4151 + 1
    QUESTION_MAX_NUM = 16891 + 1
    MAX_ATTEMPT_NUM = 5 + 1
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 111 + 1
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BATCH_SIZE = 64

    MFDKT_model = MFDKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM,
        dropout=0.2,
        user_num=MAX_USER_NUM,
        max_seq_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM
    )

    MFDKT_solver = DKTMFSolver(
        log_name='InsideLSTM/UserId+Skill/ht/SigmoidDot',
        model=MFDKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed',
        cuda='cuda:0',
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.AdamW(MFDKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    MFDKT_solver.train(epochs=50)

    MFDKT_solver.load_model(path='models_checkpoints/MFDKT/InsideLSTM/UserId+Skill/ht/SigmoidDot.pt')
    MFDKT_solver.test(MFDKT_solver.model, mode='val')
    MFDKT_solver.test(MFDKT_solver.model, mode='test')

    # print('MFDKT_solver.model.MF.embedding_layer1:\n', MFDKT_solver.model.MF.embedding_layer1)
    #
    # array = MFDKT_solver.model.MF.embedding_layer1.weight.detach().numpy()
    #
    # print(array)
    # np.savetxt('S-K', array, fmt='%.04f')
    # array = np.loadtxt('S-K')
    # print(array)

def test_PreMFDKT():
    MAX_USER_NUM = 4151 + 1
    QUESTION_MAX_NUM = 16891 + 1
    MAX_ATTEMPT_NUM = 5 + 1
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 111 + 1
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BATCH_SIZE = 64

    PreMFDKT_model = PreMFDKT(
        vocab_size=2 * SKILL_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM,
        dropout=0.2,
        user_num=MAX_USER_NUM,
        max_seq_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM
    )

    PreMFDKT_solver = PreMFDKTSolver(
        log_name='InsideLSTM/UserId+Skill/ht/SigmoidDot',
        model=PreMFDKT_model,
        models_checkpoints_dir='./models_checkpoints',
        tensorboard_log_dir='./tensorboard_logs',
        data_path='data/skill_builder_data_corrected_preprocessed',
        cuda='cuda:0',
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.AdamW(PreMFDKT_model.parameters(), lr=0.001),
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
        num_workers=1
    )

    PreMFDKT_solver.pre_train(epochs=6)
    # PreMFDKT_solver.train(epochs=50)

    PreMFDKT_solver.load_model(path='models_checkpoints/PreMFDKT/InsideLSTM/UserId+Skill/ht/SigmoidDot.pt')
    PreMFDKT_solver.test(PreMFDKT_solver.model, mode='val')
    PreMFDKT_solver.test(PreMFDKT_solver.model, mode='test')

    # print('MFDKT_solver.model.MF.embedding_layer1:\n', MFDKT_solver.model.MF.embedding_layer1)
    #
    # array = MFDKT_solver.model.MF.embedding_layer1.weight.detach().numpy()
    #
    # print(array)
    # np.savetxt('S-K', array, fmt='%.04f')
    # array = np.loadtxt('S-K')
    # print(array)

if __name__ == '__main__':
    setup_seed(41)

    # test_Baseline_DKT()

    # test_MFDKT()

    test_PreMFDKT()