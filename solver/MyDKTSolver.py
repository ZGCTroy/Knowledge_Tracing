import math
import os
import time

import sklearn
import torch
from torch import utils

from Dataset.old_version.Assistment import Assistment09


class MyDKTSolver():
    def __init__(self, model, embedding_model, data_path, device, batch_size, optimizer, model_checkpoints_dir,
                 model_name, use_pretrained_embedding, max_sequence_len):
        self.model_name = model_name
        self.model = model
        self.best_model = model
        self.embedding_model = embedding_model
        self.model_checkpoints_dir = model_checkpoints_dir
        self.train_data_loader = []
        self.test_data_loader = []
        self.val_data_loader = []
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = []
        self.use_pretrained_embedding = use_pretrained_embedding
        self.max_sequence_len = max_sequence_len

        self.load_data()

    def load_data(self):

        dataset = Assistment09(path=self.data_path, max_seq_len=self.max_sequence_len)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_size = int(0.8 * len(train_dataset))
        val_size = len(dataset) - train_size - test_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.test_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def train_one_epoch(self, cur_epoch):
        self.model.train()
        total_loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()

        batch = 0
        log_interval = 5

        for data in self.train_data_loader:

            self.optimizer.zero_grad()

            question_sequence = data['question_sequence']
            correctness_sequence = data['correct_sequence']
            skill_sequence = data['skill_sequence']
            query_question = data['query_question']
            query_skill = data['query_skill']

            output = self.model(skill_sequence, query_question, use_pretrained_embedding=False)

            loss = criterion(output.view(-1), last_is_correct.view(-1))
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    cur_epoch, batch, len(self.train_data_loader.dataset) // self.batch_size,
                    self.scheduler.get_last_lr()[0],
                                      elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            batch += 1

    def train(self, epochs, load_model_path=""):
        best_val_loss = float("inf")
        best_auc = float("inf")
        best_acc = float("inf")

        if load_model_path:
            self.load_model(path=load_model_path)
            best_val_loss, best_auc, best_acc = self.evaluate(self.model, data_loader=self.val_data_loader)
            self.best_model = self.model

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        print('start training')

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            self.train_one_epoch(epoch)

            val_loss, auc, acc = self.evaluate(self.model, data_loader=self.val_data_loader)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | auc {:8.2f} | acc{:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                        val_loss, math.exp(val_loss), auc, acc))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_auc = auc
                best_acc = acc
                self.best_model = self.model
                self.save_model(path=self.model_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

            self.scheduler.step()

    def evaluate(self, model, data_loader):

        model.eval()
        total_loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        src_mask = self.embedding_model.generate_square_subsequent_mask(self.batch_size).to(self.device)

        with torch.no_grad():
            for data in data_loader:
                question_sequence = data['question_sequence']
                last_response_pos = data['last_pos']
                last_is_correct = data['last_correct']
                correct_sequence = data['correct_sequence']
                query_question = data['query_question']

                if self.use_pretrained_embedding:
                    if question_sequence.size(0) != self.batch_size:
                        src_mask = self.embedding_model.generate_square_subsequent_mask(question_sequence.size(0)).to(
                            self.device)
                    embedding_vector1, embedding_vector2, embedding_vector3, = self.embedding_model.get_embedding_vector(
                        question_sequence, src_mask)
                    embedding_vector = embedding_vector1 + embedding_vector2 + embedding_vector3
                    embedded_query_question = embedding_vector[:, -1, :]
                    output = model(embedding_vector[:, :-2, :], embedded_query_question, use_pretrained_embedding=True)
                else:
                    output = model(question_sequence[:, -2], query_question, use_pretrained_embedding=False)

                # output = (batch_size, 1), representing the probability of correctly answering the qt
                loss = criterion(output.view(-1), last_is_correct.view(-1))
                total_loss += loss

                auc = sklearn.metrics.roc_auc_score(output.view(-1), last_is_correct.view(-1))
                acc = sklearn.metrics.accuracy_score(output.view(-1), last_is_correct.view(-1))

        return total_loss / (len(data_loader.dataset) - 1), auc, acc

    def test(self, model):
        test_loss, auc, acc = self.evaluate(model, self.test_data_loader)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | auc {:5.2f} | acc {:5.2f}'.format(
            test_loss, math.exp(test_loss), auc, acc))
        print('=' * 89)

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Read model in {} successfully\n'.format(path))
        else:
            print('Cannot find {}, use the initial model\n'.format(path))

    def save_model(self, path):
        print('New model is better, start saving ......')
        torch.save(self.model.state_dict(), path)
        print('Save model in {} successfully\n'.format(path))

# if self.use_pretrained_embedding:
#     if question_sequence.size(0) != self.batch_size:
#         src_mask = self.embedding_model.generate_square_subsequent_mask(question_sequence.size(0)).to(
#             self.device)
#     embedding_vector1, embedding_vector2, embedding_vector3, = self.embedding_model.get_embedding_vector(
#         question_sequence, src_mask)
#     embedding_vector = embedding_vector1 + embedding_vector2 + embedding_vector3
#     embedded_query_question = embedding_vector[:, -1, :]
#     output = self.model(embedding_vector[:, :-2, :], embedded_query_question, use_pretrained_embedding=True)
# else:



