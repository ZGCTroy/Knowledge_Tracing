import numpy as np
import pandas as pd
import os


def strList_to_list(strList, type='int'):
    l = strList.strip('[],')
    l = l.split(',')
    if type == 'int':
        l = [int(i) for i in l]
    elif type == 'float':
        l = [float(i) for i in l]
    elif type == 'bool':
        l = [int(i) for i in l]
    return l


def PathSim(A, B):
    A_num = A.shape[0]
    B_num = B.shape[0]

    fenzi = 2 * np.matmul(B, A.T)

    B_mol = np.sum(np.power(B, 2), axis=1).reshape(-1, 1)
    A_mol = np.sum(np.power(A, 2), axis=1).reshape(1, -1)
    B_mol_repeat = np.repeat(a=B_mol, repeats=A_num, axis=1)
    A_mol_repeat = np.repeat(a=A_mol, repeats=B_num, axis=0)
    eps = 1e-5
    AB_mol = A_mol_repeat + B_mol_repeat + eps
    similarity = np.true_divide(fenzi, AB_mol)

    return similarity


def softmax(A):
    B = np.exp(A)
    C = B / np.sum(B, axis=1).reshape(-1, 1)
    return C


def cal_similarity(root_dir, filename):

    val_df = pd.read_csv(os.path.join(root_dir, filename) + '_val.csv')
    val_df = val_df.sort_values(by=['user_id'])
    print(val_df.head(5))
    val_df = val_df.reset_index(drop=True)
    val_skill_states = list(val_df['skill_states'])
    val_skill_states = [strList_to_list(i, type='float') for i in val_skill_states]
    val_skill_states = np.array(val_skill_states)
    val_skill_states_mask = list(val_df['skill_states_mask'])
    val_skill_states_mask = [strList_to_list(i, type='bool') for i in val_skill_states_mask]
    val_skill_states_mask = np.array(val_skill_states_mask)
    val_skill_states = val_skill_states * val_skill_states_mask
    val_num = val_skill_states.shape[0]

    train_df = pd.read_csv(os.path.join(root_dir, filename) + '_train.csv')
    train_df = train_df.sort_values(by=['user_id'])
    train_df = train_df.reset_index(drop=True)
    train_skill_states = list(train_df['skill_states'])
    train_skill_states = [strList_to_list(i, type='float') for i in train_skill_states]
    train_skill_states = np.array(train_skill_states)
    train_skill_states_mask = list(train_df['skill_states_mask'])
    train_skill_states_mask = [strList_to_list(i, type='bool') for i in train_skill_states_mask]
    train_skill_states_mask = np.array(train_skill_states_mask)
    train_skill_states = train_skill_states * train_skill_states_mask
    train_num = train_skill_states.shape[0]

    test_df = pd.read_csv(os.path.join(root_dir, filename) + '_test.csv')
    test_df = test_df.sort_values(by=['user_id'])
    test_df = test_df.reset_index(drop=True)
    test_skill_states = list(test_df['skill_states'])
    test_skill_states = [strList_to_list(i, type='float') for i in test_skill_states]
    test_skill_states = np.array(test_skill_states)
    test_skill_states_mask = list(test_df['skill_states_mask'])
    test_skill_states_mask = [strList_to_list(i, type='bool') for i in test_skill_states_mask]
    test_skill_states_mask = np.array(test_skill_states_mask)
    test_skill_states = test_skill_states * test_skill_states_mask
    test_num = test_skill_states.shape[0]

    train_train_similarity = PathSim(train_skill_states, train_skill_states)
    index = np.argsort(-train_train_similarity, axis=1)
    sorted_similarity = np.take_along_axis(train_train_similarity, index, axis=1)
    train_user_id_sequence = np.repeat(np.array(train_df['user_id']).reshape(1, -1), repeats=train_num, axis=0)
    sorted_user_id = np.take_along_axis(train_user_id_sequence, index, axis=1)
    sorted_user_id = [str(i) for i in sorted_user_id.tolist()]
    sorted_similarity = [str(i) for i in sorted_similarity.tolist()]
    train_df['similar_user_id_in_train'] = sorted_user_id
    train_df['ranked_similarity_in_train'] = sorted_similarity
    train_df.to_csv(
        os.path.join(root_dir, filename + '_train.csv'),
        mode='w',
        index=False
    )

    val_train_similarity = PathSim(train_skill_states, val_skill_states)
    index = np.argsort(-val_train_similarity, axis=1)
    sorted_similarity = np.take_along_axis(val_train_similarity, index, axis=1)
    train_user_id_sequence = np.repeat(np.array(train_df['user_id']).reshape(1, -1), repeats=val_num, axis=0)
    sorted_user_id = np.take_along_axis(train_user_id_sequence, index, axis=1)
    sorted_user_id = [str(i) for i in sorted_user_id.tolist()]
    sorted_similarity = [str(i) for i in sorted_similarity.tolist()]
    val_df['similar_user_id_in_train'] = sorted_user_id
    val_df['ranked_similarity_in_train'] = sorted_similarity
    val_df.to_csv(
        os.path.join(root_dir, filename + '_val.csv'),
        mode='w',
        index=False
    )

    test_train_similarity = PathSim(train_skill_states, test_skill_states)
    index = np.argsort(-test_train_similarity, axis=1)
    sorted_similarity = np.take_along_axis(test_train_similarity, index, axis=1)
    train_user_id_sequence = np.repeat(np.array(train_df['user_id']).reshape(1, -1), repeats=test_num, axis=0)
    sorted_user_id = np.take_along_axis(train_user_id_sequence, index, axis=1)
    sorted_user_id = [str(i) for i in sorted_user_id.tolist()]
    sorted_similarity = [str(i) for i in sorted_similarity.tolist()]
    test_df['similar_user_id_in_train'] = sorted_user_id
    test_df['ranked_similarity_in_train'] = sorted_similarity
    test_df.to_csv(
        os.path.join(root_dir, filename + '_test.csv'),
        mode='w',
        index=False
    )
