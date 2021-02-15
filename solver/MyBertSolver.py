import math
import time

import os
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter

from solver.Solver import Solver


class MyBertSolver(Solver):

    def __init__(self, model, models_checkpoints_dir, tensorboard_log_dir, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num):
        super(MyBertSolver, self).__init__(
            model=model,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            cuda=cuda,
            models_checkpoints_dir=models_checkpoints_dir,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        self.skill_num = skill_num

    def pre_train(self, model, log_name, optimizer_info, epochs=5, step_size=1,
                  gamma=0.95):
        writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_log_dir, log_name))

        for i, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            model, train_loss = self.pre_train_one_epoch(model=model, optimizer_info=optimizer_info, cur_epoch=epoch)

            print('Pretraining | end of epoch {:3d} | time: {:5.2f}s | lr: {:02.4f} | train loss {:5.4f} | '
                  'train ppl {:8.4f} |'.format(epoch,
                                               (time.time() - epoch_start_time), optimizer_info['lr'],
                                               train_loss, math.exp(train_loss)))
            self.save_model(model, path=self.models_checkpoints_dir + '/' + log_name + '.pt')

            writer.add_scalar('LOSS/train', train_loss, epoch)

            for i, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch)

            if epoch % step_size == 0:
                optimizer_info['lr'] = optimizer_info['lr'] * gamma

        return model

    def pre_train_one_epoch(self, model, optimizer_info, cur_epoch=1):

        model = model.to(self.device)
        model.train()
        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 1

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_info['lr'],
            weight_decay=optimizer_info['weight_decay']
        )

        for data in self.data_loader['train']:

            label = data['skill_id_sequence'].transpose(0, 1)
            input = data['problem_id_sequence'].transpose(0, 1)
            mask = data['label_mask'].transpose(0, 1)
            cur_batch_size = label.size()[0]

            output = model(
                src=input
            ).cpu()

            label = torch.masked_select(input=label, mask=mask)
            output = torch.masked_select(
                input=output,
                mask=torch.repeat_interleave(
                    mask.unsqueeze(-1),
                    repeats=self.skill_num,
                    dim=-1
                )
            )

            output = output.view(-1, self.skill_num)

            loss = torch.nn.CrossEntropyLoss()(
                output,
                label-1
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * cur_batch_size

            if batch_id % log_interval == 0 and batch_id > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'ms/batch {:5.2f} | '
                      'loss {:5.3f} | ppl {:8.3f}'.format(
                    cur_epoch, batch_id, len(self.data_loader['train'].dataset) // self.batch_size,
                                         elapsed * 1000 / log_interval,
                    loss.item(), math.exp(loss.item())))

            start_time = time.time()
            batch_id += 1

        return model, total_loss / (len(self.data_loader['train'].dataset) - 1)

    def run_one_epoch(self, model, optimizer_info={}, cur_epoch=1, mode='', freeze=False):
        model = model.to(self.device)
        if mode == 'train':
            model.train()
            optimizer = torch.optim.AdamW(
                # [
                #     {'params': model.encoder.parameters(), 'lr': optimizer_info['lr']},
                #     {'params': model.LSTM.parameters(), 'lr': optimizer_info['lr']},
                #     {'params': model.decoder.parameters(), 'lr': optimizer_info['lr']},
                #     {'params': model.MF.parameters(), 'lr': optimizer_info['lr']},
                # ],
                model.parameters(),
                lr=optimizer_info['lr'],
                weight_decay=optimizer_info['weight_decay']
            )
            if freeze:
                model.embedding_layer.eval()
                for param in model.embedding_layer.parameters():
                    param.requires_grad = False
        else:
            model.eval()

        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 2
        total_prediction = []
        total_label = []
        total_output = []

        for data in self.data_loader[mode]:

            input = data['problem_id_sequence']

            label = data['next_correctness_sequence']
            query = data['next_problem_id_sequence']

            cur_batch_size = label.size()[0]

            output = model(
                input=input.to(self.device),
                target_id=query.to(self.device),
            ).cpu()

            output = torch.masked_select(input=output, mask=data['label_mask'])
            label = torch.masked_select(input=label, mask=data['label_mask'])

            loss = torch.nn.BCELoss()(
                output,
                label
            )
            total_loss += loss.item() * cur_batch_size

            prediction = torch.where(output > 0.5, 1, 0)
            total_prediction.extend(prediction.view(-1).detach().numpy())
            total_label.extend(label.view(-1).detach().numpy())
            total_output.extend(output.view(-1).detach().numpy())

            # 防止梯度爆炸的梯度截断，梯度超过0.5就截断
            if mode == 'train':
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0 and batch_id > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                          'loss {:5.3f} | ppl {:8.3f}'.format(
                        cur_epoch, batch_id, len(self.data_loader[mode].dataset) // self.batch_size,
                                             elapsed * 1000 / log_interval,
                        loss.item(), math.exp(loss.item())))

                start_time = time.time()

            batch_id += 1

        auc = sklearn.metrics.roc_auc_score(total_label, total_output)
        acc = sklearn.metrics.accuracy_score(total_label, total_prediction)

        return model, total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
