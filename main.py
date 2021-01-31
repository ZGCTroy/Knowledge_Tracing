import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from network.DKT import DKT
from network.MFDKT import MFDKT
from solver.DKTSolver import DKTSolver
from solver.MFDKTSolver import MFDKTSolver


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    os.environ['PYTHONHASHSEED'] = str(seed)


def test_Baseline_DKT(log_name):
    model = DKT(
        skill_num=SKILL_NUM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM + 1,
        dropout=DROP_OUT,
    )

    solver = DKTSolver(
        model=model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=CUDA,
        batch_size=BATCH_SIZE,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
    )

    # TODO 4 : Train
    solver.load_data(
        path='data/skill_builder_data_corrected_preprocessed',
        dataset_type='Assistment09'
    )

    optimizer = ''
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer = torch.optim.AdamW(solver.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_TYPE == 'SGD':
        torch.optim.SGD(solver.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer=optimizer,
        freezeMF=FREEZE_MF,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # TODO 5: Test
    with torch.no_grad():
        solver.model = solver.load_model(
            solver.model,
            path=solver.models_checkpoints_dir + '/' + log_name +'/' + solver.local_time + '.pt'
        )
        best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.best_model, mode='val')
        best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.best_model, mode='test')

        writer = SummaryWriter(log_dir='./tensorboard_logs/hparam_log/DKT')
        writer.add_hparams(
            hparam_dict={
                'model/name': 'DKT',
                'model/freeze MF': 'None',
                'model/pretrain MF': 'None',
                'model/the approach of combining MF': 'None',
                'model/extended input': 'None',
                'model/combine with': 'None',
                'model/hidden Dim': HIDDEN_DIM,
                'model/embedding Dim': EMBEDDING_DIM,
                'model/dropout': DROP_OUT,
                'model/num of linear layer': 1,
                'optimizer/type': OPTIMIZER_TYPE,
                'optimizer/start lr': LR,
                'optimizer/weight decay': WEIGHT_DECAY,
                'schedular/step size': STEP_SIZE,
                'schedular/gamma': GAMMA,
                'batch size': BATCH_SIZE,
            },
            metric_dict={
                'test auc': best_test_auc,
                'test acc': best_test_acc,
                'test loss': best_test_loss
            }
        )


def test_MF(log_name):
    # TODO 1: Set the model
    model = MFDKT(
        user_num=MAX_USER_NUM,
        skill_num=SKILL_NUM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM + 1,
        dropout=DROP_OUT,
        max_seq_len=MAX_SEQUENCE_LEN
    )

    solver = MFDKTSolver(
        model=model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=CUDA,
        batch_size=BATCH_SIZE,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
    )

    # TODO 3 : Pretrain MF
    solver.load_data(
        path='data/skill_builder_data_corrected_preprocessed',
        dataset_type='Assistment09'
    )
    solver.pre_train(
        model=solver.model.MF,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer=torch.optim.AdamW(solver.model.MF.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    )

def test_MFDKT(log_name):
    # TODO 1: Set the model
    model = MFDKT(
        user_num=MAX_USER_NUM,
        skill_num=SKILL_NUM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=SKILL_NUM + 1,
        dropout=DROP_OUT,
        max_seq_len=MAX_SEQUENCE_LEN
    )

    solver = MFDKTSolver(
        model=model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=CUDA,
        batch_size=BATCH_SIZE,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
    )

    # TODO 3 : Pretrain
    if PRETRAIN_MF:
        solver.load_data(
            path='data/skill_builder_data_corrected_preprocessed',
            dataset_type='Assistment09'
        )

        solver.pre_train(
            model=solver.model.MF,
            log_name='MFDKT/SkillLevel/UserId+SkillId/MF',
            epochs=100,
            optimizer=torch.optim.AdamW(solver.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    # TODO 4 : Train
    if USE_PRETRAINED_MF:
        solver.model.MF = solver.load_model(
            solver.model.MF,
            path='models_checkpoints/MFDKT/SkillLevel/UserId+SkillId/MF_multi_dim.pt'
        )

    solver.load_data(
        path='data/skill_builder_data_corrected_preprocessed',
        dataset_type='Assistment09'
    )

    optimizer = ''
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer = torch.optim.AdamW(solver.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_TYPE == 'SGD':
        optimizer = torch.optim.SGD(solver.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer=optimizer,
        freezeMF=FREEZE_MF,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # TODO 5: Test and Save Results
    with torch.no_grad():
        solver.model = solver.load_model(
            solver.model,
            path=solver.models_checkpoints_dir + '/' + log_name +'/' + solver.local_time + '.pt'
        )
        best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.best_model, mode='val')
        best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.best_model, mode='test')

        writer = SummaryWriter(log_dir='./tensorboard_logs/hparam_log/MFDKT')
        writer.add_hparams(
            hparam_dict={
                'model/name': 'MFDKT',
                'model/freeze MF': str(FREEZE_MF),
                'model/pretrain MF': str(USE_PRETRAINED_MF),
                'model/the approach of combining MF': 'linear layer',
                'model/extended input': 'et (multi dim)',
                'model/combine with': 'ht',
                'model/hidden Dim': HIDDEN_DIM,
                'model/embedding Dim': EMBEDDING_DIM,
                'model/dropout': DROP_OUT,
                'model/num of linear layer':1,
                'optimizer/type': OPTIMIZER_TYPE,
                'optimizer/start lr': LR,
                'optimizer/weight decay': WEIGHT_DECAY,
                'schedular/step size': STEP_SIZE,
                'schedular/gamma': GAMMA,
                'batch size': BATCH_SIZE,
            },
            metric_dict={
                'test auc': best_test_auc,
                'test acc': best_test_acc,
                'test loss': best_test_loss
            }
        )


# dataset param
MAX_USER_NUM = 4151
QUESTION_MAX_NUM = 16891
MAX_ATTEMPT_NUM = 5
MAX_SEQUENCE_LEN = 200
SKILL_NUM = 111

# solver param
MODELS_CHECKPOINTS_DIR = './models_checkpoints'
TENSORBOARD_LOG_DIR = './tensorboard_logs'

# model param
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
DROP_OUT = 0.4

# train param
OPTIMIZER_TYPE = 'AdamW'
FREEZE_MF = False
PRETRAIN_MF = False
USE_PRETRAINED_MF = True
BATCH_SIZE = 256
LR = 0.01
CUDA = 'cpu'
WEIGHT_DECAY = 0.1
MOMENTUM = 0.9
EPOCHS = 200
PATIENCE = EPOCHS // 20
STEP_SIZE = 1
GAMMA = 0.95


if __name__ == '__main__':
    # TODO 1: fix the random seed for reproduction
    setup_seed(41)

    # # TODO 2: Preprocess the data
    # from PreProcess import run
    # run()

    # TODO 3: test
    # test_Baseline_DKT(log_name='DKT/SkillLevel/Baseline')
    # test_MF(log_name='MFDKT/SkillLevel/UserId+SkillId/MF_multi_dim')
    test_MFDKT(log_name='MFDKT/SkillLevel/UserId+SkillId/MFDKT')
