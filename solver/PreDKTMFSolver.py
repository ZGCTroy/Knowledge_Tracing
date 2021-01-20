import math
import time

import sklearn.metrics
import torch

from solver.Solver import Solver


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

    def pre_train(self, epochs=5):
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

    def pre_train_one_epoch(self, model, cur_epoch=1):
        model.train()
        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 5

        for data in self.data_loader['train']:

            self.optimizer.zero_grad()

            skill_sequence = data['skill_sequence']
            user_id = data['user_id']
            user_id_sequence = torch.repeat_interleave(user_id.view(-1, 1), repeats=self.max_sequence_len, dim=1)

            hidden_vector = self.model.MF(
                user_id_sequence,
                skill_sequence,
            )

            output = model.MF.decoder(hidden_vector)

            label = data['correctness_ratio_sequence']
            loss = torch.nn.MSELoss()(
                output.view(-1, self.max_sequence_len),
                label.view(-1, self.max_sequence_len)
            )

            loss.backward()
            total_loss += loss.item()
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

        return total_loss / (len(self.data_loader['train'].dataset) - 1)

    def run_one_epoch(self, model, cur_epoch=1, mode=''):
        if mode == 'train':
            model.train()
        else:
            model.eval()

        total_loss = 0.
        start_time = time.time()
        batch_id = 0
        log_interval = 200
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

            # 防止梯度爆炸的梯度截断，梯度超过5就截断
            if mode == 'train':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
