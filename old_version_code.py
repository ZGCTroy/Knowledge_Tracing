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

