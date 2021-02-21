from __future__ import print_function, division

import math
import os
import time

import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter

from Dataset.AssistmentForBert import AssistmentForBert
from solver.Solver import Solver


class BertSolver(Solver):

    def __init__(self, model, models_checkpoints_dir, tensorboard_log_dir, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num):
        super(BertSolver, self).__init__(
            model=model,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            cuda=cuda,
            models_checkpoints_dir=models_checkpoints_dir,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        self.skill_num = skill_num

    def load_data(self, path, dataset_type, dataset_info={}, num_workers=0):
        if dataset_type in ['Assistment09', 'Assistment15', 'Assistment17']:
            train_dataset = AssistmentForBert(
                path=path + '_train.csv',
                vocab_size=dataset_info['vocab_size'],
                input_name=dataset_info['input_name'],
                label_name=dataset_info['label_name'],
                trunk_size=dataset_info['trunk_size']
            )
            val_dataset = AssistmentForBert(
                path=path + '_val.csv',
                vocab_size=dataset_info['vocab_size'],
                input_name=dataset_info['input_name'],
                label_name=dataset_info['label_name'],
                trunk_size=dataset_info['trunk_size']
            )
            test_dataset = AssistmentForBert(
                path=path + '_test.csv',
                vocab_size=dataset_info['vocab_size'],
                input_name=dataset_info['input_name'],
                label_name=dataset_info['label_name'],
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
        log_interval = 256
        total_prediction = []
        total_label = []
        total_output = []

        for data, masked_data in self.data_loader[mode]:

            input_name = model.embedding_layer.input_name

            mlm_loss = 0.
            if mode == 'train':
                masked_inputs = {}
                for name in input_name:
                    masked_inputs[name] = masked_data[name].to(self.device)

                _, _, mlm_outputs, _, _ = model(
                    src=masked_inputs,
                    src_mask=data['src_mask'].to(self.device),
                    segment_info=data['segment_info'].to(self.device)
                )


                for name in input_name:
                    mask_label = torch.masked_select(input=data[name], mask=masked_data['mask'])
                    mask_label = mask_label - 1 # id 从 0开始
                    mlm_output = mlm_outputs[name].cpu()

                    # mlm_output:（batch_size, seq_len, output_dim）--> (batch_size,mask_num, output_dim)
                    mlm_output = torch.masked_select(
                        input=mlm_output,
                        mask=torch.repeat_interleave(
                            masked_data['mask'].unsqueeze(-1),
                            repeats=mlm_output.size(-1),
                            dim=-1
                        )
                    ).view(-1, mlm_output.size(-1))

                    mlm_loss += torch.nn.CrossEntropyLoss()(input=mlm_output, target=mask_label)

            inputs = {}
            for name in input_name:
                inputs[name] = data[name].to(self.device)

            task_label = data['label'].view(-1)
            query = data['query']

            cur_batch_size = task_label.size()[0]

            _, _, _, task_output, attention_weights = model(
                src=inputs,
                segment_info=data['segment_info'].to(self.device),
                src_mask=data['src_mask'].to(self.device)
            )
            task_output = task_output.cpu()
            task_output = torch.gather(task_output, dim=1, index=query - 1).view(-1)

            task_loss = torch.nn.BCELoss()(input=task_output, target=task_label)

            loss = mlm_loss/len(input_name) + task_loss

            total_loss += loss.item() * cur_batch_size

            prediction = torch.where(task_output > 0.5, 1, 0)
            total_prediction.extend(prediction.view(-1).detach().numpy())
            total_label.extend(task_label.view(-1).detach().numpy())
            total_output.extend(task_output.view(-1).detach().numpy())

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

        if mode == 'test':
            attention_weight = attention_weights[0][0].unsqueeze(0).cpu().detach().numpy()  # (BHCW)
            attention_weight = (1 - attention_weight) * 255
            self.writer.add_image(tag='attention_weights', img_tensor=attention_weight, global_step=cur_epoch)

        auc = sklearn.metrics.roc_auc_score(total_label, total_output)
        acc = sklearn.metrics.accuracy_score(total_label, total_prediction)

        return model, total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
