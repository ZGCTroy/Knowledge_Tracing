import os
import torch
from torch.utils.tensorboard import SummaryWriter
from Dataset.data_helper import setup_seed, setup_pandas

from network.DKT import DKT
from solver.DKTSolver import DKTSolver

def test_DKT(args):
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
        solver.model, best_test_loss, best_test_auc, best_test_acc = solver.evaluate(solver.model, mode='Test')

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
                'Test auc': best_test_auc,
                'Test acc': best_test_acc,
                'Test loss': best_test_loss
            }
        )