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
        self.max_observation = T.from_numpy(observation_space.high).to(self.device)
        self.min_observation = T.from_numpy(observation_space.low).to(self.device)

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
        super(DeterministicModel, self).__init__()
        self.action_dim = action_space.shape.flatten().size
        self.state_dim = observation_space.flatten().size
        self.max_observation = T.from_numpy(observation_space.high.flatten())
        self.min_observation = T.from_numpy(observation_space.low.flatten())

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
        return mu, sigma

    def broadcast(self, state, action):
        mu, sigma = self.forward(state, action)
        probabilities = T.distributions.Normal(mu, sigma)
        state = probabilities.rsample()
        return self.max_observation * T.tanh(state)


class MBPO:

    def __init__(
        self,
        # Environment
        observation_space,
        action_space,
        # Model
        hidden_size,
        lr,
        stop_training_threshold,
        # RL Agent
        # td3,
        verbose=0
    ):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        # Environment
        self.observation_space = observation_space
        self.action_space = action_space
        self.state_dim = observation_space.sample().flatten().size
        self.action_dim = action_space.sample().flatten().size
        self.max_observation = T.from_numpy(observation_space.high.flatten()).to(self.device)
        self.min_observation = T.from_numpy(observation_space.low.flatten()).to(self.device)
        # Model
        self.env_model = DeterministicModel(
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
            observation_space, action_space, 100_000)
        self.env_buffer = ReplayBuffer(
            observation_space, action_space, buffer_size=100_000)
        # RL agent
        # self.td3 = td3
        # Dataset

    def collect_data_from_gym_env(self, env, n_timesteps, rand=True):
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

    def generate_data_from_model(self, n_timesteps, k=1, rand=True, agent=None, env=None):

        for _ in range(n_timesteps):
            # Sample data from buffer
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
                    next_obs = self.env_model(T.from_numpy(obs).reshape(
                        1, -1).to(self.device), T.from_numpy(action).reshape(1, -1).to(self.device))
                # Get reward
                reward = self.pendulum_reward(obs, action)
                # done = False
                # Store transition
                next_obs = next_obs.cpu().detach().numpy()
                done = done.cpu().detach().numpy()
                self.model_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs

    def pendulum_reward(self, obs, action):
        x, y, thetadot = obs.flatten()
        theta = np.arctan2(y, x)
        theta = ((theta + np.pi) % (2*np.pi)) - np.pi

        return -(theta ** 2 + 0.1 * thetadot ** 2 + 0.001 * (action ** 2))

    def get_data_from_replay_buffer(self, buffer, batch_size):
        observations, actions, _, next_observations, _ = buffer.sample(
            batch_size)
        inputs = (observations.to(self.device), actions.to(self.device))
        targets = next_observations.to(self.device)
        return inputs, targets

    def load_model(self, save_path):
        self.env_model.load_state_dict(T.load(save_path))

    def save_model(self, save_path):
        T.save(self.env_model.state_dict(), save_path)

    def load_dataset(self, buffer, batch_size, train_eval_split):
        inputs, targets = self.get_data_from_replay_buffer(buffer, batch_size)
        dataset = LTDataset(inputs, targets)
        train_len = int(len(dataset)*train_eval_split)
        eval_len = len(dataset) - train_len
        self.train_set, self.eval_set = T.utils.data.random_split(
            dataset, [train_len, eval_len])
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(
            dataset=self.eval_set, batch_size=1, shuffle=True)

    def train_model(self, n_epochs, broadcast=False, stop_training_threshold=2e-4, verbose=0):
        self.env_model.train(True)
        self.all_loss = []
        loss = 0
        for epoch in range(n_epochs):
            for i, (sa, target) in enumerate(self.train_loader):
                # Forward pass
                state, action = sa
                if broadcast:
                    output = self.env_model.broadcast(state, action)
                else:
                    output = self.env_model.forward(state, action)
                # Loss
                loss = self.criterion(output, target)
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
            if loss.item() < stop_training_threshold:
                return

        self.env_model.train(False)



    def eval_model(self, broadcast=False):
        all_loss = []
        with T.no_grad():
            for i, (sa, target) in enumerate(self.eval_loader):
                # Forward pass
                state, action = sa
                if broadcast:
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

    def plot_learning_curve(self):
        x = [i+1 for i in range(len(self.all_loss))]
        plt.plot(x, self.all_loss)
        plt.title('Learning Curve')

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
    mbpo = MBPO(
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
