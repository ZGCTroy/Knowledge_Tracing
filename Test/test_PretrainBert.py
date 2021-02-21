import os
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from network.BertLanguageModel.BertEncoder import BertEncoder
from network.BertLanguageModel.BertInputEmbedding import BertInputEmbedding
from network.BertLanguageModel.BertLanguageModel import BertLanguageModel, MlmDecoder, NspDecoder, TaskDecoder
from solver.PretrainBertSolver import PretrainBertSolver
from Dataset.data_helper import strList_to_list


def test_PretrainBert(args):

    # TODO 1 : get param
    if args.dataset == 'Assistment09':
        DATA_PATH = 'data/Assistment09/skill_builder_data_corrected_preprocessed'
        MAX_USER_NUM = 4151
        SKILL_NUM = 110
        QUESTION_MAX_NUM = 16891
        MAX_ATTEMPT_NUM = 5
        MAX_SEQUENCE_LEN = 200

    if args.dataset == 'Assistment17':
        DATA_PATH = 'data/Assistment17/anonymized_full_release_competition_dataset_preprocessed'
        MAX_USER_NUM = 1709
        SKILL_NUM = 102
        QUESTION_MAX_NUM = 3162
        MAX_ATTEMPT_NUM = 5
        MAX_SEQUENCE_LEN = 500

    if args.dataset == 'Assistment15':
        DATA_PATH = 'data/Assistment15/2015_100_skill_builders_main_problems_preprocessed'
        MAX_USER_NUM = 19917
        SKILL_NUM = 100
        QUESTION_MAX_NUM = 100
        MAX_ATTEMPT_NUM = 5
        MAX_SEQUENCE_LEN = 200

    vocab_size = {
        'skill_id_sequence': SKILL_NUM,
        'st+ct_sequence': SKILL_NUM * 2,
        'problem_id_sequence': QUESTION_MAX_NUM,
        'correctness_sequence': 2,
        'power_sequence': 100,
        'skill_difficulty_sequence':10,
        'problem_difficulty_sequence':100,
        'last_correctness': 2,
        'last_skill_id': SKILL_NUM,
    }

    input_name = strList_to_list(args.input_name, type='str')

    # solver param
    MODELS_CHECKPOINTS_DIR = os.path.join(args.models_checkpoints_dir, args.dataset)
    TENSORBOARD_LOG_DIR = os.path.join(args.tensorboard_log_dir, args.dataset)

    pretrain_dataset_info = {
        'vocab_size': vocab_size,
        'input_name': input_name,
        'trunk_size': args.trunk_size
    }

    dataset_info = {
        'vocab_size': vocab_size,
        'input_name': input_name,
        'query_name': args.query_name,
        'label_name': args.label_name,
        'trunk_size': args.trunk_size
    }

    print('start')
    # TODO 1: Set the model

    bert_input_embedding_layer = BertInputEmbedding(
        input_vocab_size=vocab_size,
        input_name=input_name,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        use_combine_linear=args.use_combine_linear
    )

    mlm_decoder = torch.nn.ModuleDict()
    for name in input_name:
        mlm_decoder[name] = MlmDecoder(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=vocab_size[name],
            dropout=args.dropout
        )

    bert_language_model = BertLanguageModel(
        embedding_layer=bert_input_embedding_layer,
        bert_encoder=BertEncoder(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.num_heads,
            dropout=args.dropout
        ),
        nsp_decoder=NspDecoder(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=2,
            dropout=args.dropout
        ),
        mlm_decoder=mlm_decoder,
        task_decoder=TaskDecoder(
            embedding_dim=args.embedding_dim,
            hidden_dim = args.hidden_dim,
            output_dim=SKILL_NUM,
            dropout=args.dropout
        )
    )

    print(bert_language_model)

    solver = PretrainBertSolver(
        model=bert_language_model,
        models_checkpoints_dir=MODELS_CHECKPOINTS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        cuda=args.cuda,
        batch_size=args.batch_size,
        max_sequence_len=2 * args.trunk_size + 3,
        skill_num=SKILL_NUM
    )


    # TODO 3 : Pretrain
    if args.pretrain:
        print('pretrain')

        solver.load_data(
            path=DATA_PATH,
            dataset_type=args.dataset,
            dataset_info=pretrain_dataset_info,
            pretrain=True
        )

        solver.model = solver.pre_train(
            model=solver.model,
            log_name=args.log_name,
            epochs=args.epochs,
            optimizer_info={
                'name': args.optimizer,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'step_size': args.step_size,
                'gamma': args.gamma
            },
        )
    else:

        # TODO 4 : Train
        print('train')
        if args.use_pretrained_model:
            solver.model = solver.load_model(
                solver.model,
                path=os.path.join(MODELS_CHECKPOINTS_DIR, args.log_name) + '/PretrainedBert.pt'
            )

        solver.load_data(
            path=DATA_PATH,
            dataset_type=args.dataset,
            dataset_info=dataset_info,
            pretrain=False
        )

        solver.model = solver.train(
            model=solver.model,
            log_name=args.log_name,
            epochs=args.epochs,
            optimizer_info={
                'name': args.optimizer,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'step_size': args.step_size,
                'gamma': args.gamma
            },
            patience=args.patience,
        )

        # TODO 5: Test and Save Results
        with torch.no_grad():
            solver.model = solver.load_model(
                solver.model,
                path=solver.models_checkpoints_dir + '/' + args.log_name + '/' + solver.local_time + '.pt'
            )
            solver.load_data(
                path=DATA_PATH,
                dataset_type=args.dataset,
                dataset_info=dataset_info,
                pretrain=False
            )
            solver.model, best_val_loss, best_val_auc, best_val_acc = solver.run_one_epoch(solver.model, mode='val')
            solver.model, best_test_loss, best_test_auc, best_test_acc = solver.run_one_epoch(solver.model, mode='test')

            writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, 'hparam/Bert'))
            writer.add_hparams(
                hparam_dict={
                    'model/name': 'Bert',
                    'model/use pretrain': str(args.use_pretrained_model),
                    'model/hidden Dim': args.hidden_dim,
                    'model/embedding Dim': args.embedding_dim,
                    'model/dropout': args.dropout,
                    'model/num of TransformerEncoder Layer': args.num_layers,
                    'model/num of multi-head': args.num_heads,
                    'model/use_combine_linear': str(args.use_combine_linear),
                    'dataset/type': args.dataset,
                    'dataset/pretrain_input_name': args.input_name,
                    'dataset/pretrain_label_name': args.label_name,
                    'dataset/train_input_name': args.input_name,
                    'dataset/train_query_name': args.query_name,
                    'dataset/train_label_name': args.label_name,
                    'dataset/trunk_size': args.trunk_size,
                    'optimizer/type': args.optimizer,
                    'optimizer/start lr': args.lr,
                    'optimizer/weight decay': args.weight_decay,
                    'schedular/step size': args.step_size,
                    'schedular/gamma': args.gamma,
                    'batch size': args.batch_size,
                    'log_name': args.log_name
                },
                metric_dict={
                    'Test auc': best_test_auc,
                    'Test acc': best_test_acc,
                    'Test loss': best_test_loss
                }
            )
