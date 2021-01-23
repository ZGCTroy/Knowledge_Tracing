import math
import time

import sklearn.metrics
import torch

from solver.Solver import Solver
from torch.utils.tensorboard import SummaryWriter


class PreMFDKTSolver(Solver):

    def __init__(self, model, log_name, data_path, models_checkpoints_dir, tensorboard_log_dir, optimizer, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num, num_workers=1):
        super(PreMFDKTSolver, self).__init__(
            model=model,
            log_name=log_name,
            data_path=data_path,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            cuda=cuda,
            models_checkpoints_dir=models_checkpoints_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            optimizer=optimizer,
            num_workers=num_workers,
        )
        self.skill_num = skill_num

    def pre_train(self, epochs=5):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if 'bn' not in name:
                self.writer.add_histogram(name, param, 0)

        best_val_loss, best_val_auc, best_val_acc = self.run_one_epoch(self.model, mode='val')

        print('=' * 89)
        print(
            'Initial Val Loss: {:.4f} | AUC: {:.4f} | ACC: {:.4f}\n'.format(best_val_loss, best_val_auc, best_val_acc))
        print('=' * 89)

        print('start training')

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train_loss = self.pre_train_one_epoch(self.model, cur_epoch=epoch)

            print('Pretraining | end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                  'train ppl {:8.4f} |'.format(epoch,
                                               (time.time() - epoch_start_time),
                                               train_loss, math.exp(train_loss)))
            self.save_model(path=self.models_checkpoints_dir + '/' + self.log_name + '.pt')

            val_loss, val_auc, val_acc = self.run_one_epoch(self.model, mode='val')
            test_loss, test_auc, test_acc = self.run_one_epoch(self.model, mode='test')

            print('-' * 89)

            print('Pretraining | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
                  'valid ppl {:8.4f} | val AUC {:.5f} | val ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                              val_loss, math.exp(val_loss), val_auc,
                                                                              val_acc))
            print('Pretraining || end of epoch {:3d} | time: {:5.2f}s | test loss {:5.4f} | '
                  'test ppl {:8.4f} | test AUC {:.5f} | test ACC{:.5f}'.format(epoch, (time.time() - epoch_start_time),
                                                                               test_loss, math.exp(test_loss), test_auc,
                                                                               test_acc))
            print('-' * 89)

            self.writer.add_scalar('ACC/val', val_acc, epoch)
            self.writer.add_scalar('ACC/test', test_acc, epoch)

            self.writer.add_scalar('AUC/val', val_auc, epoch)
            self.writer.add_scalar('AUC/test', test_auc, epoch)

            self.writer.add_scalar('LOSS/train', train_loss, epoch)
            self.writer.add_scalar('LOSS/val', val_loss, epoch)
            self.writer.add_scalar('LOSS/test', test_loss, epoch)

            for i, (name, param) in enumerate(self.model.named_parameters()):
                if 'bn' not in name:
                    self.writer.add_histogram(name, param, epoch)

    def pre_train_one_epoch(self, model, cur_epoch=1):
        model = model.to(self.device)
        model.train()
        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 200

        for data in self.data_loader['train']:

            self.optimizer.zero_grad()

            input1 = torch.repeat_interleave(data['user_id'].view(-1, 1), repeats=self.max_sequence_len, dim=1)
            input2 = data['skill_id_sequence']
            label = data['same_skill_correctness_ratio_sequence']
            cur_batch_size = label.size()[0]

            output = self.model.MF(
                user_id_sequence = input1.to(self.device),
                skill_id_sequence = input2.to(self.device),
            ).cpu()

            loss = torch.nn.MSELoss()(
                output.view(-1, self.max_sequence_len),
                label.view(-1, self.max_sequence_len)
            )

            loss.backward()
            total_loss += loss.item() * cur_batch_size
            self.optimizer.step()

            if batch_id % log_interval == 0 and batch_id > 0:

                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.3f} | ppl {:8.3f}'.format(
                    cur_epoch, batch_id, len(self.data_loader['train'].dataset) // self.batch_size,
                    self.scheduler.get_last_lr()[0],
                                         elapsed * 1000 / log_interval,
                    loss.item(), math.exp(loss.item())))

            start_time = time.time()
            batch_id += 1

        return total_loss / (len(self.data_loader['train'].dataset) - 1)

    def run_one_epoch(self, model, cur_epoch=1, mode=''):
        model = model.to(self.device)
        if mode == 'train':
            model.train()
        else:
            model.eval()

        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 200
        total_prediction = []
        total_label = []
        total_output = []

        for data in self.data_loader[mode]:

            self.optimizer.zero_grad()

            input1 = torch.where(data['correctness_sequence'] == 1, data['skill_id_sequence'] + self.skill_num,
                                 data['skill_id_sequence'])
            input2 = torch.repeat_interleave(data['user_id'].view(-1, 1), repeats=self.max_sequence_len, dim=1)
            input3 = data['skill_id_sequence']
            label = data['query_correctness']
            query = data['query_skill_id']
            cur_batch_size = label.size()[0]

            output = self.model(
                main_input = input1.to(self.device),  # skill_id_sequence + correctness_sequence
                target_id = query.to(self.device),
                user_id_sequence = input2.to(self.device),  # user_id_sequence
                skill_id_sequence = input3.to(self.device),  # skill_id_sequence
            ).cpu()

            loss = torch.nn.BCELoss()(
                output,
                label
            )
            total_loss += loss.item() * cur_batch_size

            prediction = torch.where(output > 0.5, 1, 0)
            total_prediction.extend(prediction.squeeze(-1).detach().numpy())
            total_label.extend(label.squeeze(-1).detach().numpy())
            total_output.extend(output.squeeze(-1).detach().numpy())

            # 防止梯度爆炸的梯度截断，梯度超过0.5就截断
            if mode == 'train':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                loss.backward()
                self.optimizer.step()

                if batch_id % log_interval == 0 and batch_id > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.4f} | ms/batch {:5.2f} | '
                          'loss {:5.3f} | ppl {:8.3f}'.format(
                        cur_epoch, batch_id, len(self.data_loader[mode].dataset) // self.batch_size,
                        self.scheduler.get_last_lr()[0],
                                             elapsed * 1000 / log_interval,
                        loss.item(), math.exp(loss.item())))

                start_time = time.time()

            batch_id += 1

        auc = sklearn.metrics.roc_auc_score(total_label, total_output)
        acc = sklearn.metrics.accuracy_score(total_label, total_prediction)

        return total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
