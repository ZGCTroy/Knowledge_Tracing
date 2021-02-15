from __future__ import print_function, division

import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from Dataset.Assistment_splitSequence import Assistment_splitSequence
from Dataset.Assistment import Assistment
from Dataset.AssistmentBert import AssistmentForBertPrtrain

class Solver():
    def __init__(self, model, models_checkpoints_dir, tensorboard_log_dir, max_sequence_len=200, cuda='cpu', batch_size=32):
        self.model = model
        self.best_model = model
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size

        self.models_checkpoints_dir = models_checkpoints_dir
        if not os.path.exists(self.models_checkpoints_dir):
            os.makedirs(self.models_checkpoints_dir)

        self.tensorboard_log_dir = tensorboard_log_dir
        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)

        self.cuda = cuda
        self.device = torch.device(self.cuda if torch.cuda.is_available() else "cpu")

        self.data_loader = {
            'train': [],
            'val': [],
            'test': []
        }

        self.local_time = str(time.asctime(time.localtime(time.time())))


    def load_data(self, path, dataset_type, split_sequence = False, pretrain=False, dataset_info={}, num_workers=0):
        if pretrain:
            if dataset_type in ['Assistment09', 'Assistment15', 'Assistment17']:
                train_dataset = AssistmentForBertPrtrain(
                    path=path + '_train.csv',
                    max_seq_len=dataset_info['max_seq_len'],
                    input_vocab_size=dataset_info['input_vocab_size'],
                    label_vocab_size=dataset_info['label_vocab_size'],
                    input_name=dataset_info['input_name'],
                    label_name=dataset_info['label_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                val_dataset = AssistmentForBertPrtrain(
                    path=path + '_val.csv',
                    max_seq_len=dataset_info['max_seq_len'],
                    input_vocab_size=dataset_info['input_vocab_size'],
                    label_vocab_size=dataset_info['label_vocab_size'],
                    input_name=dataset_info['input_name'],
                    label_name=dataset_info['label_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                test_dataset = AssistmentForBertPrtrain(
                    path=path + '_test.csv',
                    max_seq_len=dataset_info['max_seq_len'],
                    input_vocab_size=dataset_info['input_vocab_size'],
                    label_vocab_size=dataset_info['label_vocab_size'],
                    input_name=dataset_info['input_name'],
                    label_name=dataset_info['label_name'],
                    trunk_size=dataset_info['trunk_size']
                )

        else:
            if dataset_type in ['Assistment09','Assistment15', 'Assistment17']:
                if split_sequence:
                    train_dataset = Assistment_splitSequence(path=path + '.csv', max_seq_len=self.max_sequence_len, mode='train')
                    val_dataset = Assistment_splitSequence(path=path + '.csv', max_seq_len=self.max_sequence_len, mode='val')
                    test_dataset = Assistment_splitSequence(path=path + '.csv', max_seq_len=self.max_sequence_len, mode='test')
                else:
                    train_dataset = Assistment(path=path + '_train.csv', max_seq_len=self.max_sequence_len)
                    val_dataset = Assistment(path=path + '_val.csv', max_seq_len=self.max_sequence_len)
                    test_dataset = Assistment(path=path + '_test.csv', max_seq_len=self.max_sequence_len)


        self.data_loader['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.data_loader['val'] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    def run_one_epoch(self, model, optimizer='', cur_epoch=1, mode='', freezeMF=False):
        raise NotImplementedError

    def train(self, model, log_name, epochs, optimizer_info, freeze=False, patience = 5, step_size = 1,
        gamma = 0.95):

        log_dir = self.tensorboard_log_dir + '/' + log_name +'/' + self.local_time
        writer = SummaryWriter(log_dir=log_dir)

        for i, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)

        with torch.no_grad():
            model, best_val_loss, best_val_auc, best_val_acc = self.run_one_epoch(model=model,mode='val')
            model, best_test_loss, best_test_auc, best_test_acc = self.run_one_epoch(model=model,mode='test')

        print('=' * 89)
        print(
            'Initial Val Loss: {:.4f} | AUC: {:.4f} | ACC: {:.4f}'.format(best_val_loss, best_val_auc, best_val_acc))
        print(
            'Initial Test Loss: {:.4f} | AUC: {:.4f} | ACC: {:.4f}'.format(best_test_loss, best_test_auc,
                                                                             best_test_acc))
        print('=' * 89,'\n')

        print('start training')

        cur_patience = patience

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            model, train_loss, train_auc, train_acc = self.run_one_epoch(
                model=model,
                optimizer_info=optimizer_info,
                cur_epoch=epoch,
                mode='train',
                freeze=freeze
            )

            with torch.no_grad():
                model, val_loss, val_auc, val_acc = self.run_one_epoch(model, mode='val')
                model, test_loss, test_auc, test_acc = self.run_one_epoch(model, mode='test')

            print()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | lr: {:02.4f} | train loss {:5.2f} | '
                  'train ppl {:8.2f} | train AUC {:.5f} | train ACC{:.5f}'.format(epoch,
                                                                                  (time.time() - epoch_start_time),
                                                                                  optimizer_info['lr'],
                                                                                  train_loss, math.exp(train_loss),
                                                                                  train_auc, train_acc))
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | val AUC {:.5f} | val ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                              val_loss, math.exp(val_loss), val_auc,
                                                                              val_acc))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f} | test AUC {:.5f} | test ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                               test_loss, math.exp(test_loss), test_auc,
                                                                               test_acc))
            print('-' * 89)

            writer.add_scalar('ACC/train', train_acc, epoch)
            writer.add_scalar('ACC/val', val_acc, epoch)
            writer.add_scalar('ACC/test', test_acc, epoch)

            writer.add_scalar('AUC/train', train_auc, epoch)
            writer.add_scalar('AUC/val', val_auc, epoch)
            writer.add_scalar('AUC/test', test_auc, epoch)

            writer.add_scalar('LOSS/train', train_loss, epoch)
            writer.add_scalar('LOSS/val', val_loss, epoch)
            writer.add_scalar('LOSS/test', test_loss, epoch)

            writer.add_scalar('Learning Rate', optimizer_info['lr'], epoch)

            for i, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch)

            if val_auc > best_val_auc:
                cur_patience = patience
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_val_acc = val_acc

                self.save_model(model, path=self.models_checkpoints_dir + '/' + log_name+'/' + self.local_time + '.pt')
            else:
                cur_patience -= 1
                if cur_patience == 0:
                    print('Early Stop\n\n')
                    writer.close()
                    break

            if epoch % step_size == 0:
                optimizer_info['lr'] = optimizer_info['lr'] * gamma

        writer.close()
        return model

    def evaluate(self, model, mode):

        model, loss, auc, acc = self.run_one_epoch(model, mode=mode)
        print('=' * 89)
        print('{}| {} | loss {:5.4f} | ppl {:8.4f} | auc {:5.4f} | acc {:5.4f}'.format(
            model.model_name, mode, loss, math.exp(loss), auc, acc))
        print('=' * 89)

        return model, loss, auc, acc

    def save_model(self, model, path):
        print('New model is better, start saving ......')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        else:
            if os.path.exists(path):
                os.remove(path)
        torch.save(model.state_dict(), path)
        print('Save model in {} successfully\n'.format(path))

    def load_model(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print('Read model in {} successfully\n'.format(path))
        else:
            print('Cannot find {}, use the initial model\n'.format(path))
        return model
