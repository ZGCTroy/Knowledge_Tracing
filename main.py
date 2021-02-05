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
        path=DATA_PATH,
        dataset_type=DATASET_TYPE
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
                'model/extended vector': 'None',
                'model/extended information': 'None',
                'model/combine with': 'None',
                'model/hidden Dim': HIDDEN_DIM,
                'model/embedding Dim': EMBEDDING_DIM,
                'model/dropout': DROP_OUT,
                'model/num of linear layer': 1,
                'dataset/type': DATASET_TYPE,
                'dataset/max seq len': MAX_SEQUENCE_LEN,
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
        path=DATA_PATH,
        dataset_type=DATASET_TYPE
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
            path=DATA_PATH,
            dataset_type=DATASET_TYPE
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
            path=os.path.join(MODELS_CHECKPOINTS_DIR,'MFDKT/SkillLevel/UserId+SkillId/MF.pt')
        )

    solver.load_data(
        path=DATA_PATH,
        dataset_type=DATASET_TYPE
    )

    optimizer = ''
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer = torch.optim.AdamW(
            [
                {'params': solver.model.encoder.parameters(), 'lr': LR},
                {'params': solver.model.LSTM.parameters(), 'lr': LR},
                {'params': solver.model.decoder.parameters(), 'lr': LR},
                {'params': solver.model.MF.parameters(), 'lr': LR/10}
            ],
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
    if OPTIMIZER_TYPE == 'SGD':
        optimizer = torch.optim.SGD(
            [
                {'params':solver.model.encoder.parameters(),'lr':LR},
                {'params':solver.model.lSTM.parameters(),'lr':LR},
                {'params': solver.model.decoder.parameters(), 'lr': LR},
                {'params': solver.model.MF.parameters(), 'lr': LR/10}
            ],
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM
        )

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
                'model/extended vector': 'multi dim user skill embedding vector',
                'model/extended information': 'skill accuracy',
                'model/combine with': 'ht',
                'model/hidden Dim': HIDDEN_DIM,
                'model/embedding Dim': EMBEDDING_DIM,
                'model/dropout': DROP_OUT,
                'model/num of linear layer':1,
                'dataset/type':DATASET_TYPE,
                'dataset/max seq len':MAX_SEQUENCE_LEN,
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






# model param
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
DROP_OUT = 0.4

# train param
OPTIMIZER_TYPE = 'AdamW'
FREEZE_MF = True
PRETRAIN_MF = False
USE_PRETRAINED_MF = True
BATCH_SIZE = 256
LR = 0.005
CUDA = 'cpu'
WEIGHT_DECAY = 0.1
MOMENTUM = 0.9
EPOCHS = 200
PATIENCE = EPOCHS // 20
STEP_SIZE = 1
GAMMA = 0.95

# dataset param
DATASET_TYPE = 'Assistment09'
if DATASET_TYPE == 'Assistment09':
    DATA_PATH = 'data/Assistment09/skill_builder_data_corrected_preprocessed'
    MAX_USER_NUM = 4151
    SKILL_NUM = 110
    QUESTION_MAX_NUM = 16891
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 200
    MAX_SEQUENCE_LEN = 400

if DATASET_TYPE == 'Assistment17':
    DATA_PATH = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed'
    MAX_USER_NUM = 1709
    SKILL_NUM = 102
    QUESTION_MAX_NUM = 16891
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 500
    STEP_SIZE = 3

if DATASET_TYPE == 'Assistment15':
    DATA_PATH = 'data/Assistment15/2015_100_skill_builders_main_problems_preprocessed'
    MAX_USER_NUM = 19917
    SKILL_NUM = 100
    QUESTION_MAX_NUM = 100
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 200
    STEP_SIZE = 1

# solver param
MODELS_CHECKPOINTS_DIR = os.path.join('./models_checkpoints',DATASET_TYPE)
TENSORBOARD_LOG_DIR = os.path.join('./tensorboard_logs',DATASET_TYPE)

if __name__ == '__main__':
    # TODO 1: fix the random seed for reproduction
    setup_seed(41)

    # # TODO 2: Preprocess the data
    # from PreProcess import run
    # run()

    # TODO 3: test
    test_Baseline_DKT(log_name='DKT/SkillLevel/Baseline')
    # test_MF(log_name='MFDKT/SkillLevel/UserId+SkillId/MF')
    # test_MFDKT(log_name='MFDKT/SkillLevel/UserId+SkillId/MFDKT')
