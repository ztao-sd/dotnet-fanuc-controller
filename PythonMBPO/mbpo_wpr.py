import pandas as pd
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import onnx
import gym

from utils import ReplayBuffer
from td3 import *
from callback import *


class LTDataset(Dataset):

    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets
        # self.n_samples=inputs.shape[0]
        self.n_samples = inputs[0].shape[0]

        self.transform = transform

    def __getitem__(self, index):

        states, actions = self.inputs

        sample = (states[index], actions[index]), self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class DeterministicModel(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size):
        super(DeterministicModel, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.action_dim = action_space.sample().flatten().size
        self.state_dim = observation_space.sample().flatten().size
        self.max_observation = T.from_numpy(
            observation_space.high).to(self.device)
        self.min_observation = T.from_numpy(
            observation_space.low).to(self.device)

        # Q1 architecture
        self.l1 = nn.Linear(self.action_dim + self.state_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], self.state_dim)

    def forward(self, state, action):
        sa = T.cat([state, action], 1)
        sa = F.relu(self.l1(sa))
        sa = F.relu(self.l2(sa))
        sa = self.l3(sa)
        return self.max_observation * T.tanh(sa)


class GaussianModel(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size):
        super(GaussianModel, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.action_dim = action_space.sample().flatten().size
        self.state_dim = observation_space.sample().flatten().size
        self.max_observation = T.from_numpy(
            observation_space.high.flatten()).to(self.device)
        self.min_observation = T.from_numpy(
            observation_space.low.flatten()).to(self.device)

        # Q1 architecture
        self.l1 = nn.Linear(self.action_dim + self.state_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3a = nn.Linear(hidden_size[1], self.state_dim)
        self.l3b = nn.Linear(hidden_size[1], self.state_dim)

    def forward(self, state, action):
        sa = T.cat([state, action], 1)
        out = F.relu(self.l1(sa))
        out = F.relu(self.l2(out))
        mu = self.l3a(out)
        sigma = self.l3b(out)
        sigma = T.clamp(sigma, min=1e-6, max=1e5)
        return mu, sigma

    def broadcast(self, state, action):
        mu, sigma = self.forward(state, action)
        probabilities = T.distributions.Normal(mu, sigma)
        state = probabilities.rsample()
        #return self.max_observation * T.tanh(state)
        return T.clamp(state, self.min_observation, self.max_observation)


class MBPOWPR:

    pose_file_name = 'LineTrackPose.csv'
    error_file_name = 'LineTrackError.csv'
    mbpo_file_name = 'LineTrackMbpoControl.csv'
    iteration_prefix = 'iteration_'

    def __init__(
        self,
        # Environment
        observation_space,
        action_space,
        # Model
        hidden_size,
        lr,
        stop_training_threshold,
        env_buffer_size=100_000,
        model_buffer_size=100_000,
        gaussian=False,
        verbose=False
    ):
        # Device
        self.verbose = verbose
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Environment
        self.observation_space = observation_space
        self.action_space = action_space
        self.state_dim = observation_space.sample().flatten().size
        self.action_dim = action_space.sample().flatten().size
        self.max_observation = T.from_numpy(
            observation_space.high.flatten()).to(self.device)
        self.min_observation = T.from_numpy(
            observation_space.low.flatten()).to(self.device)
        self.max_action = T.from_numpy(
            action_space.high.flatten()).to(self.device)
        self.min_action = T.from_numpy(
            action_space.low.flatten()).to(self.device)

        # Model
        self.gaussian = gaussian
        if not self.gaussian:
            self.env_model = DeterministicModel(
                self.observation_space, self.action_space, hidden_size).to(self.device)
        else:
            self.env_model = GaussianModel(
                self.observation_space, self.action_space, hidden_size).to(self.device)
        self.optimizer = T.optim.Adam(self.env_model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.stop_training_threshold = stop_training_threshold
        self.train_set = None
        self.eval_set = None
        self.train_loader = None
        self.eval_loader = None
        self.all_loss = []

        # Buffer
        self.model_buffer = ReplayBuffer(
            observation_space, action_space, model_buffer_size)
        self.env_buffer = ReplayBuffer(
            observation_space, action_space, env_buffer_size)
        # RL agent
        # self.td3 = td3
        # Dataset

    #region Predictive Model

    def load_model(self, save_path):
        self.env_model.load_state_dict(T.load(save_path))

    def save_model(self, save_path):
        T.save(self.env_model.state_dict(), save_path)

    def likelyhood_loss(self, state, action, target):
        mu, sigma = self.env_model.forward(state, action)
        probabilities = T.distributions.Normal(mu, sigma)
        loss = -T.sum(probabilities.log_prob(target))
        # loss = -T.prod(probabilities.log_prob(target))
        return loss

    def train_model(self, n_epochs, stop_training_threshold=2e-4, verbose=0):
        self.env_model.train(True)
        self.all_loss = []
        loss = 0
        for epoch in range(n_epochs):
            for i, (sa, target) in enumerate(self.train_loader):
                # Forward pass
                state, action = sa
                # Loss
                if not self.gaussian:
                    output = self.env_model.forward(state, action)
                    loss = self.criterion(output, target)
                else:
                    loss = self.likelyhood_loss(state, action, target)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Eval data
                self.all_loss.append(loss.item())

            if verbose:
                if (epoch+1) % 20 == 0:
                    print(f"Epoch: {epoch+1}/{n_epochs} | Loss: {loss}")

            # Termination
            if not self.gaussian and loss.item() < stop_training_threshold:
                break

        self.env_model.train(False)

    def eval_model(self):
        all_loss = []
        with T.no_grad():
            for i, (sa, target) in enumerate(self.eval_loader):
                # Forward pass
                state, action = sa
                if self.gaussian:
                    output = self.env_model.broadcast(state, action)
                else:
                    output = self.env_model.forward(state, action)
                # Loss
                loss = self.criterion(output, target)
                all_loss.append(loss.item())
        pass
        # Plot validation curve
        x = [i+1 for i in range(len(all_loss))]
        running_avg = np.zeros(len(all_loss))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(all_loss[max(0, i-10):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Learning Curve')
        print(f"Avg loss: {np.mean(all_loss):2f}")

    def get_data_from_replay_buffer(self, buffer, batch_size):
        observations, actions, _, next_observations, _ = buffer.sample(
            batch_size)
        inputs = (observations.to(self.device), actions.to(self.device))
        targets = next_observations.to(self.device)
        return inputs, targets

    def normalize_input(self, states, actions):
        norm_states = []
        norm_actions = []
        for state, action in zip(states, actions):
            state = T.mul(state-self.min_observation, 2 /
                          (self.max_observation-self.min_observation)) - 1
            action = T.mul(action-self.min_action, 2 /
                           (self.max_action-self.min_action)) - 1
            norm_states.append(state.reshape(1, -1))
            norm_actions.append(action.reshape(1, -1))

        return T.concat(norm_states, dim=0), T.concat(norm_actions, dim=0)

    def load_dataset(self, buffer, batch_size, train_eval_split):
        inputs, targets = self.get_data_from_replay_buffer(buffer, batch_size)

        # Normalize
        # states, actions = inputs
        # inputs = self.normalize_input(states, actions)
        dataset = LTDataset(inputs, targets)
        train_len = int(len(dataset)*train_eval_split)
        eval_len = len(dataset) - train_len
        self.train_set, self.eval_set = T.utils.data.random_split(
            dataset, [train_len, eval_len])
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            dataset=self.eval_set, batch_size=1, shuffle=True)

    #endregion

    #region Environment Data

    def collect_data_from_gym_env(self, env, n_timesteps, rand=True):
        """
        Collect state transitions from env and store to self.env_buffer
        """
        observation = env.reset()
        for _ in range(n_timesteps):
            if rand:
                action = env.action_space.sample()
            else:
                action = self.td3.exploration_action(observation)
            # Perform action
            next_observation, reward, done, _ = env.step(action)
            # Store transition
            self.env_buffer.add(observation, action, reward,
                                next_observation, done)
            observation = next_observation
            # Reset on episode end
            if done:
                observation, done = env.reset(), False

    def normalize_error_control(self, error_array, control_array, obs_space, action_space):
        # max_obs = self.max_observation.cpu().detach().numpy()
        # min_obs = self.min_observation.cpu().detach().numpy()
        # max_act = self.max_action.cpu().detach().numpy()
        # min_act = self.min_action.cpu().detach().numpy()
        max_obs = obs_space.high
        min_obs = obs_space.low
        max_act = action_space.high
        min_act = action_space.low

        norm_errors = []
        norm_controls = []
        for error, control in zip(error_array, control_array):
            norm_error = np.array([
                np.interp(xq, [min, max], [-1, 1])
                for xq, min, max in zip(error, min_obs, max_obs)
            ])
            norm_errors.append(norm_error.reshape(1, -1))
            norm_control = np.array([
                np.interp(xq, [min, max], [-1, 1])
                for xq, min, max in zip(control, min_act, max_act)
            ])
            norm_controls.append(norm_control.reshape(1, -1))

        return np.concatenate(norm_errors, axis=0, dtype=np.float32), np.concatenate(norm_controls, axis=0, dtype=np.float32)

    def env_data_from_files(self, data_dirs, obs_space, action_space):
        """
        Collect state transitions from csv files
        """
        for dir_ in data_dirs:
            if MBPOWPR.iteration_prefix in dir_:
                pose_path = os.path.join(dir_, MBPOWPR.pose_file_name)
                error_path = os.path.join(dir_, MBPOWPR.error_file_name)
                control_path = os.path.join(dir_, MBPOWPR.mbpo_file_name)
                pose_array = np.genfromtxt(pose_path, delimiter=',',
                                            skip_header=1, dtype=np.float32)
                error_array = np.genfromtxt(error_path, delimiter=',',
                                            skip_header=1, dtype=np.float32)
                control_array = np.genfromtxt(control_path, delimiter=',',
                                              skip_header=1, dtype=np.float32)
                pose_rows = []
                error_rows = []
                control_rows = []
                for pose_row, error_row, control_row in zip(pose_array, error_array, control_array):
                    if not np.isnan(np.sum(error_row)) and not np.isnan(np.sum(control_row)) and not np.isnan(np.sum(pose_row)):
                        # Normalize data
                        pose_rows.append(pose_row[4:].reshape(1, -1))
                        error_rows.append(error_row[4:].reshape(1, -1))
                        control_rows.append(control_row[4:].reshape(1, -1))
                pose_array = np.concatenate(pose_rows, axis=0)
                error_array = np.concatenate(error_rows, axis=0)
                control_array = np.concatenate(control_rows, axis=0)
                # Combine pose and error
                obs_array = np.concatenate([pose_array, error_array], axis=1)
                obs_array, control_array = self.normalize_error_control(
                    obs_array, control_array, obs_space, action_space)

                self.prev_obs = None
                for i in range(error_array.shape[0]-1):
                    observation = obs_array[i]
                    action = control_array[i]
                    reward = self.line_track_reward_v2(observation)
                    next_observation = obs_array[i+1]
                    if i == obs_array.shape[0]-2:
                        done = True
                    else:
                        done = False
                    self.env_buffer.add(observation, action, reward,
                                        next_observation, done)

    #endregion

    #region Model Data

    def generate_data_from_model(self, n_timesteps, k=1, rand=True, agent=None, env=None, reward_function=None):

        if reward_function is None:
            reward_function = self.pendulum_reward

        for _ in range(n_timesteps):
            # Sample data from environment buffer
            observations, _, _, next_observations, done = self.env_buffer.sample(
                1)
            if np.random.choice(2):
                obs = observations.cpu().detach().numpy()
            else:
                obs = next_observations.cpu().detach().numpy()
            for _ in range(k):
                # Get new actions from next_observations
                if rand:
                    action = self.action_space.sample()
                else:
                    action = agent.exploration_action(obs)
                # Perform action
                with T.no_grad():
                    if not self.gaussian:
                        next_obs = self.env_model.forward(T.from_numpy(obs).reshape(
                            1, -1).to(self.device), T.from_numpy(action).reshape(1, -1).to(self.device))
                    else:
                        next_obs = self.env_model.broadcast(T.from_numpy(obs).reshape(
                            1, -1).to(self.device), T.from_numpy(action).reshape(1, -1).to(self.device))
                # Get reward
                reward = reward_function(obs, action)
                # done = False
                # Store transition
                next_obs = next_obs.cpu().detach().numpy()
                done = done.cpu().detach().numpy()
                self.model_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs

    #endregion

    #region ONNX

    def export_to_onnx(self, save_file):
        dummy_input = T.randn(
            self.state_dim + self.action_dim, device=self.device)
        input_names = ["state", "action"]
        T.onnx.export(self.env_model, dummy_input, save_file, verbose=True)

    def test_onnx_model(self, save_file, verbose=0):
        onnx_model = onnx.load(save_file)
        if onnx.checker.check_model(onnx_model):
            if verbose:
                print("Model checked!")
                print(onnx.helper.printable_graph(onnx_model.graph))
            return True
        else:
            if verbose:
                print("Model not good!")
            return False

    def onnx_model_inference(self, onnx_model, sa):
        pass

    #endregion

    #region Pendulum

    def pendulum_reward(self, obs, action):
        x, y, thetadot = obs.flatten()
        theta = np.arctan2(y, x)
        theta = ((theta + np.pi) % (2*np.pi)) - np.pi

        return -(theta ** 2 + 0.1 * thetadot ** 2 + 0.001 * (action ** 2))

    #endregion

    #region Fanuc Line Tracking

    def line_track_reward(self, obs, action=None):
        reward = -np.sum(np.abs(obs))
        return reward

    def line_track_reward_v2(self, obs, action=None):

        r=1.5
        # path error penalty
        # factor = 0.05 * 20
        # if self.prev_obs is not None:
        #     if factor < np.amax(obs[1:4]) * 20:
        #         factor = np.amax(obs[1:4]) * 20
        r1 = (np.abs(obs[3]) + np.abs(obs[4]) + np.abs(obs[5])) * 10
        #r1 = (np.abs(obs[6])*r + np.abs(obs[7])*r + np.abs(obs[8])*r+np.abs(obs[9]) + np.abs(obs[10]) + np.abs(obs[11])) * 10
        # r1 = 10 * np.sqrt((np.sum(obs[1:4]**2)))
        # if np.abs(obs[3]) > 0.7:
        #     r1 += 2
        # if np.abs(obs[5]) > 7/20:
        #     r1 += 100

        # path oscillation penalty
        r2 = 0
        # if self.prev_obs is not None:
        #     r2 = np.abs(obs[1]-self.prev_obs[1]) + np.abs(
        #         obs[2] - self.prev_obs[2]) + np.abs(obs[3] - self.prev_obs[3])
        #     r2 *= 5

        reward = r1 + r2
        self.prev_obs = obs

        return -reward

    #endregion

    #region Plotting

    def plot_learning_curve(self, save_path=None):
        fig, ax = plt.subplots()
        x = [i+1 for i in range(len(self.all_loss))]
        ax.plot(x, self.all_loss)
        ax.set_title('Learning Curve')
        if save_path is not None:
            fig.savefig(save_path)

    #endregion


if __name__ == "__main__":
    """Create environment"""
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')
    observation_space = env.observation_space
    action_space = env.action_space
    observation_shape = observation_space.shape
    action_dim = action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    device = T.device("cuda" if T.cuda.is_available() else 'cpu')
    """RL agent"""
    # Create model
    hidden_size_actor = [60, 50]
    hidden_size_critic = [120, 100]
    # Create TD3 agent
    agent = TD3(
        # Environment
        observation_space=observation_space,
        action_space=action_space,
        max_action=max_action,
        min_action=min_action,
        # Model
        hidden_size_actor=hidden_size_actor,
        hidden_size_critic=hidden_size_critic,
        # Training hypers
        learn_delay=100,
        policy_delay=2,
        gradient_steps=1,
        batch_size=100,
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        tau=5e-3,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        exploration_noise=0.1,
        verbose=0
    )
    # Create callback
    eval_callback = EvalCallBack(
        eval_env,
        n_eval_episodes=5,
        eval_freq=1_000,  # timesteps
        log_path=None,
        best_model_save_path=None,
        verbose=1
    )
    eval_callback.init_callback(env, agent)
    """MBPO"""
    # Hyperparameters (Model learning)
    model_rollout_freq = 1  # 1 per 1 timestep
    n_epochs = 500
    stop_training_threshold = 0.0002
    lr_model = 1e-3
    # Hyperparameters (MBPO)
    total_timesteps = 20_000
    gradient_steps = 1000  # 1000 per episode
    model_buffer_size = 100_000
    env_buffer_size = 50_000
    model_gen_data_size = 10_000
    horizon = 1
    warmup_steps = 2000
    # Create MBPO object
    mbpo = MBPOWPR(
        env.observation_space,
        env.action_space,
        hidden_size=[100, 80],
        lr=lr_model,
        stop_training_threshold=stop_training_threshold,
        # td3=agent,
        verbose=0
    )

    """Learn"""
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    observation, done = env.reset(), False
    for t in range(total_timesteps):

        # Update time
        agent.time_step += 1
        episode_timesteps += 1

        # Select action
        if t < agent.learn_delay:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation)
            noise = np.random.normal(
                0, agent.exploration_noise, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)
        # Perform action
        next_observation, reward, done, _ = env.step(action)

        # Store transition in environment buffer
        mbpo.env_buffer.add(observation, action, reward,
                            next_observation, done)
        observation = next_observation
        episode_reward += reward

        # Generate data from predictive model
        # if t > 1000:
        #     mbpo.generate_data_from_model(n_timesteps=2, k=1, rand=False, agent=agent)

        # Handle done
        if done:
            # Train policy
            if t > warmup_steps:
                mbpo.generate_data_from_model(
                    n_timesteps=model_gen_data_size, k=horizon, rand=False, agent=agent)
                agent.train(mbpo.model_buffer, gradient_steps)
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            observation, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if (episode_num+1) % 1 == 0:
                # Train model
                mbpo.load_dataset(mbpo.env_buffer, mbpo.env_buffer.pos, 0.99)
                mbpo.train_model(
                    n_epochs=n_epochs, stop_training_threshold=stop_training_threshold, verbose=1)

        # Evaluation
        eval_callback.on_step()
