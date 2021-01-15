import numpy as np
import random
import torch
import os
from network.DKT import DKT
from network.MyBert import MyBertModel
from solver.MyBertSolver import MyBertSolver
from solver.MyDKTSolver import MyDKTSolver
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

    QUESTION_MAX_NUM = 200000
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LEN = 200
    SKILL_NUM = 112

    # my_bert_model = MyBertModel(
    #     vocab_size=QUESTION_MAX_NUM, # number of different problem id
    #     embedding_dim=EMBEDDING_DIM,
    #     output_dim=400, # number of different skills
    #     num_head=2,
    #     num_hidden=400,
    #     num_encoder_layers=2,
    #     dropout=0.1,
    # )
    #
    # my_bert_solver = MyBertSolver(
    #     model=my_bert_model,
    #     model_name = 'MyBert',
    #     model_checkpoints_dir= 'model_checkpoints',
    #     data_path='data/skill_builder_data_corrected_small.csv',
    #     device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
    #     batch_size=20,
    #     optimizer=torch.optim.SGD(my_bert_model.parameters(), lr=0.5),
    #     max_sequence_len=MAX_SEQUENCE_LEN
    # )

    # solver.train(
    #     epochs=30,
    #     load_model_path=""
    # )

    # my_bert_solver.load_model(path='model_checkpoints/MyBert_best_train_01_13.pt')
    # my_bert_solver.test(my_bert_solver.model)

    print('start DKT training')
    DKT_model = DKT(
        vocab_size = 2*SKILL_NUM+1,
        embedding_dim = 100,
        hidden_dim= 100,
        num_layers=1,
        output_dim = SKILL_NUM,
        dropout = 0.2
    )

    DKT_solver = DKTSolver(
        model = DKT_model,
        model_name='DKT',
        model_checkpoints_dir='model_checkpoints',
        data_path='data/skill_builder_data_corrected_preprocessed.csv',
        device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
        batch_size=64,
        optimizer=torch.optim.SGD(DKT_model.parameters(), lr=0.001),
        use_pretrained_embedding=False,
        max_sequence_len=MAX_SEQUENCE_LEN,
        skill_num = SKILL_NUM
    )
    # DKT_solver.load_model(path='model_checkpoints/DKT_best_train.pt')

    DKT_solver.train(
        epochs = 50,
        load_model_path=""
    )

    DKT_solver.load_model(path='model_checkpoints/DKT_best_train.pt')
    DKT_solver.test(DKT_solver.model)



