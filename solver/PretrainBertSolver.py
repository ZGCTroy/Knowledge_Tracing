from __future__ import print_function, division

import math
import os
import time

import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from Dataset.AssistmentForBertTrain import AssistmentForBertTrain
from Dataset.AssistmentForBertPretrain import AssistmentForBertPrtrain
from solver.Solver import Solver


class PretrainBertSolver(Solver):

    def __init__(self, model, models_checkpoints_dir, tensorboard_log_dir, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num):
        super(PretrainBertSolver, self).__init__(
            model=model,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            cuda=cuda,
            models_checkpoints_dir=models_checkpoints_dir,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        self.skill_num = skill_num

    def load_data(self, path, dataset_type, pretrain=False, dataset_info={}, num_workers=0):
        if pretrain:
            if dataset_type in ['Assistment09', 'Assistment15', 'Assistment17']:
                train_dataset = AssistmentForBertPrtrain(
                    path=path + '_train.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                val_dataset = AssistmentForBertPrtrain(
                    path=path + '_val.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                test_dataset = AssistmentForBertPrtrain(
                    path=path + '_test.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )

        else:
            if dataset_type in ['Assistment09', 'Assistment15', 'Assistment17']:
                train_dataset = AssistmentForBertTrain(
                    path=path + '_train.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                val_dataset = AssistmentForBertTrain(
                    path=path + '_val.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )
                test_dataset = AssistmentForBertTrain(
                    path=path + '_test.csv',
                    vocab_size=dataset_info['vocab_size'],
                    input_name=dataset_info['input_name'],
                    trunk_size=dataset_info['trunk_size']
                )

        self.data_loader['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.data_loader['val'] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers
        )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers
        )

    def pre_train(self, model, log_name, optimizer_info, epochs=5):
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
            self.save_model(model, path=self.models_checkpoints_dir + '/' + log_name + '/PretrainedBert.pt')

            writer.add_scalar('LOSS/train', train_loss, epoch)

            for i, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch)

            if epoch % optimizer_info['step_size'] == 0:
                optimizer_info['lr'] = optimizer_info['lr'] * optimizer_info['gamma']

        return model

    def pre_train_one_epoch(self, model, optimizer_info, cur_epoch=1):

        model = model.to(self.device)
        model.train()
        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 10

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_info['lr'],
            weight_decay=optimizer_info['weight_decay']
        )

        for data, masked_data in self.data_loader['train']:

            cur_batch_size = data['nsp_label'].size(0)

            input_name = model.embedding_layer.input_name
            masked_inputs = {}
            inputs = {}
            for name in input_name:
                inputs[name] = data[name].to(self.device)
                masked_inputs[name] = masked_data[name].to(self.device)

            embedding_output, nsp_output, mlm_outputs, task_output, _ = model(
                src=masked_inputs,
                segment_info= data['segment_info'].to(self.device),
                src_mask=data['src_mask'].to(self.device)
            )

            nsp_loss = torch.nn.CrossEntropyLoss()(
                input=nsp_output,
                target=data['nsp_label'].to(self.device).view(-1)
            )

            mlm_loss = 0.
            for name in input_name:
                mask_label = torch.masked_select(
                    input=data[name].to(self.device),
                    mask=masked_data['mask'].to(self.device)
                )
                mask_label = mask_label - 1  # id 从 0开始
                output_dim = mlm_outputs[name].size(-1)

                # mlm_output:（batch_size, seq_len, output_dim）--> (batch_size,mask_num, output_dim)
                mlm_output = torch.masked_select(
                    input=mlm_outputs[name],
                    mask=torch.repeat_interleave(
                        masked_data['mask'].unsqueeze(-1).to(self.device),
                        repeats=mlm_outputs[name].size(-1),
                        dim=-1
                    )
                )

                mlm_loss += torch.nn.CrossEntropyLoss()(
                    input=mlm_output.view(-1, output_dim),
                    target=mask_label.view(-1)
                )

            loss = nsp_loss + mlm_loss/ len(input_name)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * cur_batch_size

            if batch_id % log_interval == 0 and batch_id > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      's {:5.2f} | '
                      'loss {:5.3f} | ppl {:8.3f}'.format(
                    cur_epoch, batch_id, len(self.data_loader['train'].dataset) // self.batch_size,
                    elapsed,
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
        log_interval = 1024
        total_prediction = []
        total_label = []
        total_output = []

        for data in self.data_loader[mode]:
            cur_batch_size = data['label'].size(0)

            inputs={}
            for name in model.embedding_layer.input_name:
                inputs[name] = data[name].to(self.device)

            embedding_output, _, _, task_output, attention_weights = model(
                src=inputs,
                segment_info=data['segment_info'].to(self.device),
                src_mask=data['src_mask'].to(self.device)
            )

            # query 第 t+1 时刻 skill id的 知识状态
            task_output = torch.gather(
                task_output,
                dim=1,
                index=(data['query'] - 1).to(self.device)
            )

            task_loss = torch.nn.BCELoss()(
                input=task_output.view(cur_batch_size,-1),
                target=data['label'].to(self.device).view(cur_batch_size, -1)
            )
            loss = task_loss
            total_loss += loss.item() * cur_batch_size

            prediction = torch.where(task_output.view(-1) > 0.5, 1, 0)
            total_prediction.extend(prediction.view(-1).cpu().detach().numpy())
            total_label.extend(data['label'].view(-1).detach().numpy())
            total_output.extend(task_output.view(-1).cpu().detach().numpy())

            # 防止梯度爆炸的梯度截断，梯度超过0.5就截断
            if mode == 'train':
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0 and batch_id > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | s {:5.2f} | '
                          'loss {:5.3f} | ppl {:8.3f}'.format(
                        cur_epoch, batch_id, len(self.data_loader[mode].dataset) // self.batch_size,
                                             elapsed,
                        loss.item(), math.exp(loss.item())))

                    start_time = time.time()

            batch_id += 1

        auc = sklearn.metrics.roc_auc_score(total_label, total_output)
        acc = sklearn.metrics.accuracy_score(total_label, total_prediction)

        if mode == 'test':
            attention_weight = attention_weights[0][0].unsqueeze(0).cpu().detach().numpy()  # (HCW)
            attention_weight = (1 - attention_weight) * 255 # 0 (white) ~ 255 (black)
            self.writer.add_image(tag='attention_weights', img_tensor=attention_weight, global_step=cur_epoch)

        return model, total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
