from pickletools import optimize
from re import S
from tkinter import TOP
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnx


class LTDataset(Dataset):
    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.n_samples = inputs.shape[0]

        self.transform = transform

    def __getitem__(self, index):

        sample = self.inputs[index], self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class MLPNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden1, n_hidden2, max_error):
        super(MLPNetwork, self).__init__()
        self.max_error = max_error
        self.fc1 = nn.Linear(n_inputs, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.pred = nn.Linear(n_hidden2, n_outputs)

        # self.lnorm1 = nn.LayerNorm(n_hidden1)
        # self.lnorm2 = nn.LayerNorm(n_hidden2)

    def forward(self, input):
        out = self.fc1(input)
        # out = self.lnorm1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # out = self.lnorm2(out)
        out = F.relu(out)
        out = self.pred(out)
        return self.max_error*T.tanh(out)


class MLPGNetwork(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden1, n_hidden2):
        super(MLPGNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.mu = nn.Linear(n_hidden2, n_outputs)
        self.sigma = nn.Linear(n_hidden2, n_outputs)

        # Weight initialization
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 3e-3
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)
        self.sigma.weight.data.uniform_(-f3, f3)
        self.sigma.bias.data.uniform_(-f3, f3)

        # self.lnorm1 = nn.LayerNorm(n_hidden1)
        # self.lnorm2 = nn.LayerNorm(n_hidden2)

    def forward(self, input):
        out = self.fc1(input)
        # out = self.lnorm1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # out = self.lnorm2(out)
        out = F.relu(out)

        mu = self.mu(out)
        sigma = self.sigma(out)
        sigma = T.clamp(sigma, min=1e-9, max=1e5)

        return mu, sigma


class PNN:

    path_error_file_name = 'LineTrackError.csv'
    control_file_name = 'LineTrackControl.csv'
    iteration_prefix = 'iteration_'
    # path_error_file_name = 'E_Data.csv'
    # control_file_name = 'U_Data.csv'
    # iteration_prefix = 'Iteration_'
    min_control = -0.10
    max_control = 0.10

    def __init__(self, inputs, targets, lr, train_split, hidden_layers, max_error,
                 max_control, gaussian=False) -> None:

        # General
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.gaussian = gaussian
        # self.device = T.device('cpu')

        # Data
        self.inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.targets = T.tensor(targets, dtype=T.float).to(self.device)
        self.dataset = LTDataset(inputs, targets)

        # Dataset
        self.train_split = train_split
        train_len = int(len(self.dataset)*self.train_split)
        val_len = len(self.dataset)-train_len
        self.train_set, self.val_set = T.utils.data.random_split(
            self.dataset, [train_len, val_len])

        # Train loader
        self.batch_size = int(len(self.train_set)*1.0)
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

        # Validation loader
        self.val_loader = DataLoader(
            dataset=self.val_set, batch_size=1, shuffle=True)

        # Neural Network
        self.max_error = max_error
        self.max_control = max_control
        self.n_inputs = inputs.shape[1]
        self.n_outputs = targets.shape[1]
        if self.gaussian:
            self.model = MLPGNetwork(n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                                     n_hidden1=hidden_layers[0], n_hidden2=hidden_layers[1]).to(self.device)
        else:
            self.model = MLPNetwork(n_inputs=self.n_inputs, n_outputs=self.n_outputs, n_hidden1=hidden_layers[0],
                                    n_hidden2=hidden_layers[1], max_error=self.max_error).to(self.device)

        # Hyperparameters
        self.learning_rate = lr
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Learning curves
        self.losses = []
        self.running_losses = []

    def likelyhood_loss(self, input, target):
        mu, sigma = self.model.forward(input)
        probabilities = T.distributions.Normal(mu, sigma)
        loss = -T.sum(probabilities.log_prob(target))
        # loss = -T.prod(probabilities.log_prob(target))
        return loss

    def mse_loss(self, input, target):
        mu, sigma = self.model.forward(input)
        probabilities = T.distributions.Normal(mu, sigma)
        output = probabilities.rsample()
        output = 0.25*T.tanh(output)
        loss = self.criterion(output, target)
        return loss

    def learn(self, n_epochs):

        # Loss
        loss = 0
        running_loss = 0
        self.losses = []
        self.running_losses = []
        self.eval_losses_mean = []
        self.eval_losses_std = []
        for epoch in range(n_epochs):
            for i, (input, target) in enumerate(self.train_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model.forward(input)

                # Loss
                loss = self.criterion(output, target)
                self.losses.append(loss.item())
                running_loss = np.mean(self.losses[-10:])
                self.running_losses.append(running_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Display
                if (epoch+1) % 10 == 0:
                    print(
                        f'Epoch: {epoch+1}/{n_epochs} | Loss {loss} | Running Loss {running_loss}')
                    # Evaluate
                    self.evaluate()
                    self.eval_losses_mean.append(np.mean(self.eval_losses))
                    self.eval_losses_std.append(np.std(self.eval_losses))

    def evaluate(self):
        self.eval_losses = []
        loss = 0
        with T.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model.forward(input)
                loss = self.criterion(output, target)
                self.eval_losses.append(loss.item())

    #region Import/Export

    def save(self, save_path):
        T.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(T.load(load_path))

    def save_onnx(self, save_path):
        dummy_input = T.randn(self.n_inputs, device=self.device)
        input_names = ['prev_control']
        output_names = ['error']
        T.onnx.export(self.model, dummy_input, save_path, input_names=input_names,
                      output_names=output_names, verbose=True)

    #endregion

    #region Data IO

    @staticmethod
    def normalize_input(input_array, min_values, max_values):

        norm_inputs = []
        for input in input_array:
            norm_input = np.array([
                np.interp(xq, [min, max], [-1, 1])
                for xq, min, max in zip(input, min_values, max_values)
            ])
            norm_inputs.append(norm_input.reshape(1, -1))
        return np.concatenate(norm_inputs, axis=0, dtype=np.float32)

    @staticmethod
    def load_data(top_dir):

        control_arrays = []
        error_arrays = []
        dirs = os.listdir(top_dir)
        iter_num = 0
        for dir_ in dirs:
            if PNN.iteration_prefix in dir_:
                iter_num += 1
                data_path = os.path.join(top_dir, dir_, PNN.control_file_name)
                control_array = np.genfromtxt(data_path, delimiter=',',
                                              skip_header=1, dtype=np.float32)
                control_arrays.append(control_array)
                data_path = os.path.join(
                    top_dir, dir_, PNN.path_error_file_name)
                error_array = np.genfromtxt(data_path, delimiter=',',
                                            skip_header=1, dtype=np.float32)
                error_arrays.append(error_array)
        return control_arrays, error_arrays

    @staticmethod
    def shift_data(shift, control_arrays, error_arrays):

        inputs = []
        targets = []
        for control_array, error_array in zip(control_arrays, error_arrays):

            # Input
            input = np.concatenate([
                row.reshape(1,-1) for row in control_array if not np.isnan(np.sum(row))
            ], axis=0)
            start_time = input[0][0]
            input = input[shift-1:-1, :4]
            for i in range(shift-1):
                input = np.concatenate(
                    [input, error_array[shift-i-1:-i-1, 1:4]], axis=1)
            input[:,0] = np.subtract(input[:,0], start_time)
            inputs.append(input)

            # Target
            target = np.concatenate([
                row.reshape(1,-1) for row in error_array if not np.isnan(np.sum(row))
            ], axis=0)
            targets.append(target[shift:, 1:4])

        input_array = np.concatenate(inputs, axis=0)
        target_array = np.concatenate(targets, axis=0)

        # Normalize control
        input_array = PNN.normalize_input(input_array, min_values=[0]+[PNN.min_control]*(input_array.shape[0]-1),
                                          max_values=[35] + [PNN.max_control]*(input_array.shape[0]-1))
        max = np.amax(input_array)
        min = np.amin(input_array)
        return input_array, target_array

    #endregion


class PNNPlot:

    @staticmethod
    def pose_data_plot(ax, df, subtitle, xlabel='time (sec)', ylim=[-0.10, 0.10], legend=False):
        t, x, y, z = df['time'], df['x'], df['y'], df['z']
        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.plot(t, z, label='z')
        ax.set_xlabel(xlabel)
        ax.set_title(subtitle)
        ax.set_ylim(ylim)
        if legend:
            ax.legend()


if __name__ == '__main__':

    # TOP_DIR = r'D:\Fanuc Experiments\test-0327-pcontrol\output'
    TOP_DIR = r'D:\Fanuc Experiments\stuff\ML - Python\data\dec_14\run_2'

    control_arrays, error_arrays = PNN.load_data(TOP_DIR)
    input_array, target_array = PNN.shift_data(1, control_arrays, error_arrays)
    print(input_array.shape)
    print(target_array.shape)
    print(target_array)

    pnn_kwargs = {
        'inputs': input_array,
        'targets': target_array,
        'lr': 1e-3,
        'train_split': 0.8,
        'hidden_layers': [100, 100]
    }
    pnn = PNN(**pnn_kwargs)
    save_path = os.path.join(TOP_DIR, 'mlpg_0401.pt')
    # if True:
    #     pnn.load(save_path)
    pnn.learn(n_epochs=700)
    pnn.evaluate()
    save_path = os.path.join(TOP_DIR, 'mlpg_0401.pt')
    pnn.save(save_path)
    save_path = os.path.join(TOP_DIR, 'mlpg_0401.onnx')
    pnn.save_onnx(save_path)

    # Plot learning curves
    fig, ax = plt.subplots()
    x = [i+1 for i in range(len(pnn.losses))]
    ax.plot(x, pnn.losses)
    ax.set_title('Learning Curve')

    # Plot validation mean and std
    # fig, ax = plt.subplots()
    # x = [i+1 for i in range(len(pnn.eval_losses_mean))]
    # ax.plot(x, pnn.eval_losses_mean)
    # ax.plot(x, pnn.eval_losses_std)
    # ax.set_title('Mean and Std of Valiadation Set')

    # Plot validation curve
    fig, ax = plt.subplots()
    x = [i+1 for i in range(len(pnn.eval_losses))]
    ax.plot(x, pnn.eval_losses)
    ax.set_title('Validation Curve')


plt.show()
