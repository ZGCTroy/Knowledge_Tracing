import os

import torch
from torch.utils.tensorboard import SummaryWriter

from Dataset.data_helper import setup_seed
from network.BertLanguageModel.BertEncoder import BertEncoder
from network.BertLanguageModel.BertLanguageModel import BertLanguageModel, MlmDecoder, NspDecoder, TaskDecoder
from network.DKT import DKT
from network.MFDKT import MFDKT
from solver.DKTSolver import DKTSolver
from solver.MFDKTSolver import MFDKTSolver
from solver.MyBertSolver import MyBertSolver
from solver.PretrainBertSolver import PretrainBertSolver


def test_Baseline_DKT(log_name):
    model = DKT(
        vocab_size=QUESTION_MAX_NUM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=QUESTION_MAX_NUM + 1,
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
        dataset_type=DATASET_TYPE,
        split_sequence=SPLIT_SEQUENCE
    )

    optimizer_info = {}
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer_info['name'] = 'AdamW'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY
    if OPTIMIZER_TYPE == 'SGD':
        optimizer_info['name'] = 'SGD'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY

    solver.model = solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer_info=optimizer_info,
        freezeMF=FREEZE_MF,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # TODO 5: Test
    with torch.no_grad():
        solver.model = solver.load_model(
            solver.model,
            path=solver.models_checkpoints_dir + '/' + log_name + '/' + solver.local_time + '.pt'
        )
        solver.model, best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.model, mode='val')
        solver.model, best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.model, mode='test')

        writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, 'hparam/DKT'))
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

    optimizer_info = {}
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer_info['name'] = 'AdamW'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY
    if OPTIMIZER_TYPE == 'SGD':
        optimizer_info['name'] = 'SGD'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY

    # TODO 3 : Pretrain MF
    solver.load_data(
        path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        split_sequence=SPLIT_SEQUENCE
    )
    solver.model.MF = solver.pre_train(
        model=solver.model.MF,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer_info=optimizer_info,
        step_size=STEP_SIZE,
        gamma=GAMMA
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

    optimizer_info = {}
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer_info['name'] = 'AdamW'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY
    if OPTIMIZER_TYPE == 'SGD':
        optimizer_info['name'] = 'SGD'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY

    # TODO 3 : Pretrain
    if PRETRAIN:
        solver.load_data(
            path=DATA_PATH,
            dataset_type=DATASET_TYPE,
            split_sequence=SPLIT_SEQUENCE
        )

        solver.model.MF = solver.pre_train(
            model=solver.model.MF,
            log_name='MFDKT/SkillLevel/UserId+SkillId/MF',
            epochs=100,
            optimizer_info=optimizer_info,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    solver.load_data(
        path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        split_sequence=SPLIT_SEQUENCE
    )
    # TODO 4 : Train
    if USE_PRETRAINED_MF:
        solver.model.MF = solver.load_model(
            solver.model.MF,
            path=os.path.join(MODELS_CHECKPOINTS_DIR, 'MFDKT/SkillLevel/UserId+SkillId/MF.pt')
        )
        solver.model.MF = solver.pre_train(
            model=solver.model.MF,
            log_name='test_MF',
            epochs=2,
            optimizer_info=optimizer_info,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    solver.model = solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer_info=optimizer_info,
        freezeMF=FREEZE_MF,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # TODO 5: Test and Save Results
    with torch.no_grad():
        solver.model = solver.load_model(
            solver.model,
            path=solver.models_checkpoints_dir + '/' + log_name + '/' + solver.local_time + '.pt'
        )
        solver.model, best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.model, mode='val')
        solver.model, best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.model, mode='test')

        writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, 'hparam/MFDKT'))
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


def test_MyBert(log_name):
    # TODO 1: Set the model

    model = DKT(
        vocab_size=QUESTION_MAX_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        output_dim=QUESTION_MAX_NUM + 1,
        dropout=DROP_OUT,
    )

    solver = MyBertSolver(
        model=model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=CUDA,
        batch_size=BATCH_SIZE,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM,
    )

    Bert_model = MyBert(
        vocab_size=QUESTION_MAX_NUM + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=SKILL_NUM,
        num_encoder_layers=2,
        num_head=2,
        dropout=DROP_OUT,
        max_seq_len=MAX_SEQUENCE_LEN
    )
    optimizer_info = {}
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer_info['name'] = 'AdamW'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY
    if OPTIMIZER_TYPE == 'SGD':
        optimizer_info['name'] = 'SGD'
        optimizer_info['lr'] = LR
        optimizer_info['weight_decay'] = WEIGHT_DECAY

    solver.load_data(
        path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        split_sequence=SPLIT_SEQUENCE
    )
    # TODO 3 : Pretrain

    if PRETRAIN:
        Bert_model = solver.pre_train(
            model=Bert_model,
            log_name='MyBert',
            epochs=100,
            optimizer_info=optimizer_info,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    # TODO 4 : Train
    if USE_PRETRAINED_EMBEDDING:
        Bert_model = solver.load_model(
            Bert_model,
            path=os.path.join(MODELS_CHECKPOINTS_DIR, 'MyBert.pt')
        )

        print(solver.model.embedding_layer.weight)
        print(solver.model.embedding_layer.weight.shape)
        solver.model.embedding_layer = Bert_model.embedding_layer
        print(solver.model.embedding_layer.weight)

    solver.model = solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer_info=optimizer_info,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        # freeze = FREEZE_EMBEDDING
    )
    #
    # # TODO 5: Test and Save Results
    # with torch.no_grad():
    #     solver.model = solver.load_model(
    #         solver.model,
    #         path=solver.models_checkpoints_dir + '/' + log_name + '/' + solver.local_time + '.pt'
    #     )
    #     solver.model, best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.model, mode='val')
    #     solver.model, best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.model, mode='test')
    #
    #     writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR,'hparam/MFDKT'))
    #     writer.add_hparams(
    #         hparam_dict={
    #             'model/name': 'MFDKT',
    #             'model/freeze MF': str(FREEZE_MF),
    #             'model/pretrain MF': str(USE_PRETRAINED_MF),
    #             'model/the approach of combining MF': 'linear layer',
    #             'model/extended vector': 'multi dim user skill embedding vector',
    #             'model/extended information': 'skill accuracy',
    #             'model/combine with': 'ht',
    #             'model/hidden Dim': HIDDEN_DIM,
    #             'model/embedding Dim': EMBEDDING_DIM,
    #             'model/dropout': DROP_OUT,
    #             'model/num of linear layer': 1,
    #             'dataset/type': DATASET_TYPE,
    #             'dataset/max seq len': MAX_SEQUENCE_LEN,
    #             'optimizer/type': OPTIMIZER_TYPE,
    #             'optimizer/start lr': LR,
    #             'optimizer/weight decay': WEIGHT_DECAY,
    #             'schedular/step size': STEP_SIZE,
    #             'schedular/gamma': GAMMA,
    #             'batch size': BATCH_SIZE,
    #         },
    #         metric_dict={
    #             'test auc': best_test_auc,
    #             'test acc': best_test_acc,
    #             'test loss': best_test_loss
    #         }
    #     )


def test_PretrainBert(log_name):
    print('start')
    # TODO 1: Set the model

    bert_encoder = BertEncoder(
        vocab_size=SKILL_NUM + 3 + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMERENCODER_LAYERS,
        attn_heads=NUM_HEADS,
        dropout=DROP_OUT
    )

    mlm_decoder = MlmDecoder(
        embedding_dim=EMBEDDING_DIM,
        output_dim=SKILL_NUM
    )

    nsp_decoder = NspDecoder(
        embedding_dim=EMBEDDING_DIM,
        output_dim=2
    )

    task_decoder = TaskDecoder(
        embedding_dim=EMBEDDING_DIM,
        output_dim=SKILL_NUM
    )

    bert_language_model = BertLanguageModel(
        bert_encoder=bert_encoder,
        nsp_decoder=nsp_decoder,
        mlm_decoder=mlm_decoder,
        task_decoder=task_decoder
    )

    solver = PretrainBertSolver(
        model=bert_language_model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=CUDA,
        batch_size=BATCH_SIZE,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num=SKILL_NUM
    )

    optimizer_info = {
        'name': OPTIMIZER_TYPE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY
    }

    # TODO 3 : Pretrain
    if PRETRAIN:
        print('pretrain')
        pretrain_dataset_info = {
            'input_vocab_size': SKILL_NUM,
            'label_vocab_size': SKILL_NUM,
            'input_name': 'skill_id_sequences',
            'label_name': 'skill_id_sequences',
            'trunk_size': TRUNK_SIZE
        }

        solver.load_data(
            path=DATA_PATH,
            dataset_type=DATASET_TYPE,
            split_sequence=SPLIT_SEQUENCE,
            dataset_info=pretrain_dataset_info,
            pretrain=PRETRAIN
        )

        solver.model = solver.pre_train(
            model=bert_language_model,
            log_name='BertLM',
            epochs=100,
            optimizer_info=optimizer_info,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

    # TODO 4 : Train
    # if USE_PRETRAINED_EMBEDDING:
    #     bert_language_model = solver.load_model(
    #         bert_language_model,
    #         path=os.path.join(MODELS_CHECKPOINTS_DIR, 'MyBert.pt')
    #     )
    #
    #     print(solver.model.embedding_layer.weight)
    #     print(solver.model.embedding_layer.weight.shape)
    #     solver.model.embedding_layer = bert_language_model.embedding_layer
    #     print(solver.model.embedding_layer.weight)

    dataset_info = {
        'input_vocab_size': SKILL_NUM,
        'label_vocab_size': SKILL_NUM,
        'input_name': 'skill_id_sequences',
        'query_name': 'skill_id_querys',
        'label_name': 'correctness',
        'trunk_size': TRUNK_SIZE
    }

    solver.load_data(
        path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        dataset_info=dataset_info,
        pretrain=False
    )

    print('train')
    solver.model = solver.train(
        model=solver.model,
        log_name=log_name,
        epochs=EPOCHS,
        optimizer_info=optimizer_info,
        patience=PATIENCE,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        # freeze = FREEZE_EMBEDDING
    )

    # TODO 5: Test and Save Results
    with torch.no_grad():
        solver.model = solver.load_model(
            solver.model,
            path=solver.models_checkpoints_dir + '/' + log_name + '/' + solver.local_time + '.pt'
        )
        solver.model, best_val_loss, best_val_auc, best_val_acc = solver.evaluate(solver.model, mode='val')
        solver.model, best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.model, mode='test')

        writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR,'hparam/Bert'))
        writer.add_hparams(
            hparam_dict={
                'model/name': 'Bert',
                'model/freeze': str(FREEZE_EMBEDDING),
                'model/pretrain': str(USE_PRETRAINED_EMBEDDING),
                'model/hidden Dim': HIDDEN_DIM,
                'model/embedding Dim': EMBEDDING_DIM,
                'model/dropout': DROP_OUT,
                'model/num of TransformerEncoder Layer': NUM_TRANSFORMERENCODER_LAYERS,
                'model/num of multi-head':NUM_HEADS,
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


# dataset param
DATASET_TYPE = 'Assistment09'

if DATASET_TYPE == 'Assistment09':
    DATA_PATH = 'data/Assistment09/skill_builder_data_corrected_preprocessed'
    MAX_USER_NUM = 4151
    SKILL_NUM = 110
    QUESTION_MAX_NUM = 16891
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 200

if DATASET_TYPE == 'Assistment17':
    DATA_PATH = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed'
    MAX_USER_NUM = 1709
    SKILL_NUM = 102
    QUESTION_MAX_NUM = 3162
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 500

if DATASET_TYPE == 'Assistment15':
    DATA_PATH = 'data/Assistment15/2015_100_skill_builders_main_problems_preprocessed'
    MAX_USER_NUM = 19917
    SKILL_NUM = 100
    QUESTION_MAX_NUM = 100
    MAX_ATTEMPT_NUM = 5
    MAX_SEQUENCE_LEN = 200

# model param
EMBEDDING_DIM = 240
HIDDEN_DIM = 240
DROP_OUT = 0.5

# train param
OPTIMIZER_TYPE = 'AdamW'
FREEZE_MF = True
PRETRAIN = False
USE_PRETRAINED_MF = True
FREEZE_EMBEDDING = True
USE_PRETRAINED_EMBEDDING = True
BATCH_SIZE = 256
LR = 0.0001
CUDA = 'cpu'
WEIGHT_DECAY = 0.01
MOMENTUM = 0.9
EPOCHS = 200
PATIENCE = EPOCHS // 20
STEP_SIZE = 1
GAMMA = 1.0
SPLIT_SEQUENCE = False
TRUNK_SIZE = 35
NUM_HEADS = 12
NUM_TRANSFORMERENCODER_LAYERS =6

# solver param
MODELS_CHECKPOINTS_DIR = os.path.join('./models_checkpoints', DATASET_TYPE)
TENSORBOARD_LOG_DIR = os.path.join('./tensorboard_logs', DATASET_TYPE)

if __name__ == '__main__':
    # TODO 1: fix the random seed for reproduction
    setup_seed(41)

    # # TODO 2: Preprocess the data
    # from PreProcess import run
    # run()

    # TODO 3: test
    # test_Baseline_DKT(log_name='DKT/SkillLevel/Baseline')
    # test_MF(log_name='MFDKT/SkillLevel/UserId+SkillId/MF')
    # test_MFDKT(log_name='MFDKT/SkillLevel/UserId+SkillId/MFDKT')
    # test_MyBert(log_name='MyBert')
    test_PretrainBert(log_name='BertBaseline')
