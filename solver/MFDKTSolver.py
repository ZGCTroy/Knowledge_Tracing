import math
import time

import sklearn.metrics
import torch

from solver.Solver import Solver


class MFDKTSolver(Solver):

    def __init__(self, model, log_name, data_path, models_checkpoints_dir, tensorboard_log_dir, optimizer, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num, num_workers=1):
        super(MFDKTSolver, self).__init__(
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
        # self.writer.add_graph(
        #     self.model,
        #     input_to_model=[
        #         torch.ones(size=(64, 200)).long(),
        #         torch.ones(size=(64, 1)).long(),
        #         torch.ones(size=(64, 200)).long(),
        #         torch.ones(size=(64, 200)).long(),
        #         torch.ones(size=(64, 200)).long(),
        #     ]
        # )

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
