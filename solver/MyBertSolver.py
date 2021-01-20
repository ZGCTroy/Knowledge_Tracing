import math
import os
import time

import torch
from torch import utils

from DataSet import Assistment09



# from network.DKT import DKT


class MyBertSolver():
    def __init__(self, model, data_path, device, batch_size, optimizer, model_checkpoints_dir, model_name,
                 max_sequence_len):
        self.model_name = model_name
        self.model = model
        self.best_model = model
        self.model_checkpoints_dir = model_checkpoints_dir
        self.train_data_loader = []
        self.test_data_loader = []
        self.val_data_loader = []
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optimizer
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

        self.scheduler = []

    def train_one_epoch(self, cur_epoch):
        self.model.train()  # Turn on the train mode
        total_loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        src_mask = self.model.generate_square_subsequent_mask(self.batch_size).to(self.device)

        batch = 0
        log_interval = 5

        for data in self.train_data_loader:
            question_sequence = data['question_sequence']
            skill_sequence = data['skill_sequence']
            self.optimizer.zero_grad()

            if question_sequence.size(0) != self.batch_size:
                src_mask = self.model.generate_square_subsequent_mask(question_sequence.size(0)).to(self.device)

            output = self.model(question_sequence, src_mask)
            loss = criterion(output.view(-1, self.model.output_dim), skill_sequence.view(-1))
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

        if load_model_path:
            self.load_model(path=load_model_path)
            best_val_loss = self.evaluate(self.model, data_loader=self.val_data_loader)
            self.best_model = self.model

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        print('start training')

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            self.train_one_epoch(epoch)

            val_loss = self.evaluate(self.model, data_loader=self.val_data_loader)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model
                self.save_model(path=self.model_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

            self.scheduler.step()

    def evaluate(self, model, data_loader):

        model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        src_mask = model.generate_square_subsequent_mask(self.batch_size).to(self.device)

        with torch.no_grad():
            for data in data_loader:
                question_sequence = data['question_sequence']
                skill_sequence = data['skill_sequence']
                if question_sequence.size(0) != self.batch_size:
                    src_mask = model.generate_square_subsequent_mask(question_sequence.size(0)).to(self.device)

                output = model(question_sequence, src_mask)
                loss = criterion(output.view(-1, model.output_dim), skill_sequence.view(-1))
                total_loss += len(question_sequence) * loss

        return total_loss / (len(data_loader.dataset) - 1)

    def test(self, model):
        test_loss = self.evaluate(model, self.test_data_loader)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
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
