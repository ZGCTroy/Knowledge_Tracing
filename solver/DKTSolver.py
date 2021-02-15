import math
import time

import sklearn.metrics
import torch

from solver.Solver import Solver


class DKTSolver(Solver):

    def __init__(self, model, models_checkpoints_dir, tensorboard_log_dir, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num):
        super(DKTSolver, self).__init__(
            model=model,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            cuda=cuda,
            models_checkpoints_dir=models_checkpoints_dir,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        self.skill_num = skill_num

    def run_one_epoch(self, model, optimizer_info={}, cur_epoch=1, mode='', freezeMF=True):
        model = model.to(self.device)
        if mode == 'train':
            model.train()
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.encoder.parameters(), 'lr': optimizer_info['lr']},
                    {'params': model.LSTM.parameters(), 'lr': optimizer_info['lr']},
                    {'params': model.decoder.parameters(), 'lr': optimizer_info['lr']},
                ],
                lr=optimizer_info['lr'],
                weight_decay=optimizer_info['weight_decay']
            )
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

            # SkillLevel Input
            input = torch.where(
                data['correctness_sequence'] == 1,
                data['problem_id_sequence'] + self.skill_num,
                data['problem_id_sequence']
            )
            label = data['next_correctness_sequence']
            query = data['next_problem_id_sequence']
            cur_batch_size = label.size()[0]

            output = model(
                input = input.to(self.device),
                target_id = query.to(self.device)
            ).cpu()

            output = torch.masked_select(input = output, mask = data['label_mask'])
            label = torch.masked_select(input = label, mask = data['label_mask'])

            loss = torch.nn.BCELoss()(
                output,
                label
            )

            total_loss += loss.item() * cur_batch_size

            prediction = torch.where(output > 0.5, 1., 0.)
            total_prediction.extend(prediction.view(-1).detach().numpy())
            total_label.extend(label.view(-1).detach().numpy())
            total_output.extend(output.view(-1).detach().numpy())

            # 防止梯度爆炸的梯度截断，梯度超过5就截断
            if mode == 'train':
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0 and batch_id > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'ms/batch {:5.2f} | '
                          'loss {:5.3f} | ppl {:8.3f}'.format(
                        cur_epoch, batch_id, len(self.data_loader[mode].dataset) // self.batch_size,

                                             elapsed * 1000 / log_interval,
                        loss.item(), math.exp(loss.item())))

                start_time = time.time()

            batch_id += 1

        auc = sklearn.metrics.roc_auc_score(total_label, total_output)
        acc = sklearn.metrics.accuracy_score(total_label, total_prediction)


        return model, total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
