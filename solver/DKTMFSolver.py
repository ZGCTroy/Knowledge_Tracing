import math
import time

import sklearn.metrics
import torch

from solver.Solver import Solver


class DKTMFSolver(Solver):

    def __init__(self, model, log_name,data_path, models_checkpoints_dir, tensorboard_log_dir, optimizer, cuda,
                 batch_size,
                 max_sequence_len,
                 skill_num, num_workers=1):
        super(DKTMFSolver, self).__init__(
            model=model,
            log_name = log_name,
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
        self.writer.add_graph(
            self.model,
            input_to_model=[
                torch.ones(size=(64, 200)).long(),
                torch.ones(size=(64, 1)).long(),
                torch.ones(size=(64, 200)).long(),
                torch.ones(size=(64, 200)).long(),
                torch.ones(size=(64, 200)).long(),
            ]
        )

    def run_one_epoch(self, model, cur_epoch=1, mode=''):
        if mode == 'train':
            model.train()
        else:
            model.eval()

        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 5
        total_predictions = []
        total_query_correctness = []
        total_correct_probability = []

        for data in self.data_loader[mode]:

            self.optimizer.zero_grad()

            skill_sequence = data['skill_sequence']
            question_sequence = data['question_sequence']
            correctness_sequence = data['correctness_sequence']
            query_skill = data['query_skill']
            query_question = data['query_question']
            query_correctness = data['query_correctness']
            attempt_sequence = data['attempt_sequence']
            user_id = data['user_id']
            user_id_sequence = torch.repeat_interleave(user_id.view(-1, 1), repeats=self.max_sequence_len, dim=1)

            # SkillLevel
            input = skill_sequence
            query = query_skill

            # QuestionLevel
            # input = question_sequence
            # query = query_question

            input = torch.where(correctness_sequence == 1, input + self.skill_num, input)

            correctness_probability = self.model(
                input,
                query,
                user_id_sequence,
                skill_sequence,
                attempt_sequence
            )

            loss = torch.nn.BCELoss()(
                correctness_probability,
                query_correctness
            )
            total_loss += loss.item()

            query_correctness = data['query_correctness']
            predictions = torch.where(correctness_probability > 0.5, 1, 0)
            total_predictions.extend(predictions.squeeze(-1).data.cpu().numpy())
            total_query_correctness.extend(query_correctness.squeeze(-1).data.cpu().numpy())
            total_correct_probability.extend(correctness_probability.squeeze(-1).data.cpu().numpy())

            # 防止梯度爆炸的梯度截断，梯度超过0.5就截断
            if mode == 'train':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                loss.backward()
                self.optimizer.step()

                if batch_id % log_interval == 0 and batch_id > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.4f} | ms/batch {:5.2f} | '
                          'loss {:5.3f} | ppl {:8.3f}'.format(
                        cur_epoch, batch_id, len(self.data_loader['train'].dataset) // self.batch_size,
                        self.scheduler.get_last_lr()[0],
                                             elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                start_time = time.time()

            batch_id += 1

        auc = sklearn.metrics.roc_auc_score(total_query_correctness, total_correct_probability)
        acc = sklearn.metrics.accuracy_score(total_query_correctness, total_predictions)

        return total_loss / (len(self.data_loader[mode].dataset) - 1), auc, acc
