from __future__ import print_function, division

import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from DataSet import Assistment09
import sklearn


class Solver():
    def __init__(self, model, optimizer, models_checkpoints_dir, tensorboard_log_dir,data_path = '', max_sequence_len=200, cuda='cuda:0', batch_size=32, num_workers=2):
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

        self.num_workers = num_workers

        self.cuda = cuda
        self.device = torch.device(self.cuda if torch.cuda.is_available() else "cpu")

        self.data_loader = {
            'train': [],
            'val': [],
            'test': []
        }
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        self.writer = None
        self.load_data(path=data_path)

    def load_data(self, path):

        dataset = Assistment09(path=path, max_sequence_len=self.max_sequence_len)

        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(41)
        )

        train_size = int(0.8 * len(train_dataset))
        val_size = len(dataset) - train_size - test_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(41)
        )

        self.data_loader['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.data_loader['val'] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def run_one_epoch(self, model, cur_epoch=1, mode = ''):
        raise NotImplementedError

    def train(self, epochs):
        local_time = str(time.asctime(time.localtime(time.time())))
        log_dir = self.tensorboard_log_dir + '/' + self.model.model_name + '/' + local_time
        self.writer = SummaryWriter(log_dir=log_dir)

        best_val_loss, best_val_auc, best_val_acc = self.run_one_epoch(self.model, mode = 'val')

        print('=' * 89)
        print('Initial Val Loss: {:.4f} | AUC: {:.4f} | ACC: {:.4f}\n'.format(best_val_loss, best_val_auc, best_val_acc))
        print('=' * 89)

        print('start training')

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_auc, train_acc = self.run_one_epoch(self.model, cur_epoch=epoch,mode = 'train')

            val_loss, val_auc, val_acc = self.run_one_epoch(self.model, mode = 'val')

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                  'train ppl {:8.2f} | train AUC {:.5f} | train ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                      train_loss, math.exp(val_loss), train_auc, train_acc))
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | val AUC {:.5f} | val ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                              val_loss, math.exp(val_loss), val_auc,
                                                                              val_acc))
            print('-' * 89)

            self.writer.add_scalars('ACC/train_val', {'train acc':train_acc, 'val acc':val_acc}, epoch)
            self.writer.add_scalar('ACC/train',train_acc,epoch)
            self.writer.add_scalar('ACC/val',val_acc,epoch)
            self.writer.add_scalars('AUC/train_val', {'train auc': train_auc, 'val auc': val_auc}, epoch)
            self.writer.add_scalar('AUC/train',train_auc,epoch)
            self.writer.add_scalar('AUC/val',val_auc,epoch)
            self.writer.add_scalars('LOSS/train_val', {'train loss':train_loss, 'val loss':val_loss}, epoch)
            self.writer.add_scalar('LOSS/train',train_loss,epoch)
            self.writer.add_scalar('LOSS/val',val_loss,epoch)
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if 'bn' not in name:
                    self.writer.add_histogram(name, param, epoch)

            if val_auc > best_val_auc:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_val_acc = val_acc
                self.best_model = self.model
                self.save_model(path=self.models_checkpoints_dir + '/' + self.model.model_name + local_time+'.pt')

        # self.scheduler.step()
        self.writer.close()

    def test(self, model):
        test_loss, auc, acc = self.run_one_epoch(model, mode='test')
        print('=' * 89)
        print('| test loss {:5.2f} | test ppl {:8.2f} | auc {:5.2f} | acc {:5.2f}'.format(
            test_loss, math.exp(test_loss), auc, acc))
        print('=' * 89)

    def save_model(self, path):
        print('New model is better, start saving ......')
        torch.save(self.model.state_dict(), path)
        print('Save model in {} successfully\n'.format(path))

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Read model in {} successfully\n'.format(path))
        else:
            print('Cannot find {}, use the initial model\n'.format(path))
