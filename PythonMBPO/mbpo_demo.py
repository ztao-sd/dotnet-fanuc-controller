
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import onnx
import pandas as pd

import os
import glob
import shutil

from utils import *
from td3 import *
from callback import *
from mbpo import *

if __name__ == "__main__":

    #region File system
    top_dir = r'D:\Fanuc Experiments\test-master'
    output_dir = os.path.join(top_dir, 'output')
    rl_policy_save_dir = os.path.join(output_dir, "rl_policies")
    model_save_dir = os.path.join(output_dir, "models")
    data_save_dir = os.path.join(output_dir, "data")
    os.makedirs(rl_policy_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(data_save_dir, exist_ok=True)
    #endregion

    #region Environment
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')
    observation_space = env.observation_space
    action_space = env.action_space
    observation_shape = observation_space.shape
    action_dim = action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    device = T.device("cuda" if T.cuda.is_available() else 'cpu')
    #endregion

    #region TD3
    td3_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'max_action': max_action,
        'min_action':min_action,
        'hidden_size_actor': [60, 50],
        'hidden_size_critic': [120, 100],
        'learn_delay': 100,
        'policy_delay': 2,
        'gradient_steps': 1,
        'batch_size': 100,
        'gamma': 0.99,
        'lr_actor': 1e-3,
        'lr_critic': 1e-3,
        'tau': 5e-3,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'exploration_noise': 0.1,
        'verbose': 0
    }
    td3 = TD3(**td3_args_dict)
    #endregion

    #region MBPO
    mbpo_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'hidden_size': [60, 50],
        'lr': 5e-3,
        'stop_training_threshold': 2e-4,
        'gaussian': False,
        'verbose':False
    }
    mbpo = MBPO(**mbpo_args_dict)
    # Hyperparameters (MBPO)
    n_epochs = 500
    model_rollout_freq = 1  # 1 per 1 timestep
    total_timesteps = 20_000
    gradient_steps = 1000  # 1000 per episode
    model_buffer_size = 100_000
    env_buffer_size = 50_000
    model_gen_data_size = 10_000
    horizon = 1
    warmup_steps = 2000
    #endregion

    #region Callbacks
    eval_callback = EvalCallBack(
        eval_env,
        n_eval_episodes=5,
        eval_freq=1_000,  # timesteps
        log_path=None,
        best_actor_save_path=None,
        best_critic_save_path=None,
        verbose=1
    )
    checkpoint_callback = CheckPointCallback(save_freq=5000)
    eval_callback.init_callback(env, td3, mbpo)
    checkpoint_callback.init_callback(env, td3, mbpo)
    #endregion

    #region Load Models
    continued=False
    if continued:
        try:
            model_path = os.path.join(model_save_dir, 'model_best.pt')
            mbpo.load_model(model_path)
        except Exception as e:
            print(e)

    #endregion

    #region Training
    episode_score = []
    episode_length = []
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    observation, done = env.reset(), False
    for t in range(total_timesteps):

        # Update time
        td3.time_step += 1
        episode_timesteps += 1

        # Select action
        if t < td3.learn_delay:
            action = env.action_space.sample()
        else:
            action = td3.select_action(observation)
            noise = np.random.normal(
                0, td3.exploration_noise, size=action_dim)
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
        #     mbpo.generate_data_from_model(n_timesteps=2, k=1, rand=False, agent=td3)

        # Handle done
        if done:
            # Train policy
            if t > warmup_steps:
                mbpo.generate_data_from_model(
                    n_timesteps=model_gen_data_size, k=horizon, rand=False, agent=td3)
                td3.train(mbpo.model_buffer, gradient_steps)
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            observation, done = env.reset(), False
            episode_score.append(episode_reward)
            episode_length.append(episode_timesteps)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if (episode_num+1) % 1 == 0:
                # Train model
                mbpo.load_dataset(mbpo.env_buffer, mbpo.env_buffer.pos, 0.99)
                mbpo.train_model(
                    n_epochs=n_epochs, stop_training_threshold=mbpo.stop_training_threshold, verbose=1)
                # Save model learning data
                model_learning_data_path = os.path.join(data_save_dir, f"model_learning_data_eps{episode_num}.csv")
                df = pd.DataFrame(mbpo.all_loss, columns=["Loss"])
                df.to_csv(model_learning_data_path)

        # Evaluation
        eval_callback.best_actor_save_path = os.path.join(rl_policy_save_dir, f"best_actor.pt")
        eval_callback.best_critic_save_path = os.path.join(rl_policy_save_dir, f"best_critic.pt")
        eval_callback.on_step()

        # Save checkpoints
        checkpoint_callback.model_save_dir = model_save_dir
        checkpoint_callback.data_save_dir = data_save_dir
        checkpoint_callback.on_step()

        # Save rl learning data
        if (td3.time_step+1) % 2000 == 0:
            rl_learning_data_path = os.path.join(data_save_dir, f"rl_learning_data_eps{episode_num}.csv")
            df = pd.DataFrame(list(zip(episode_score, episode_length)), columns=["Eps Score", "Eps Length"])
            df.to_csv(rl_learning_data_path)
    #endregion
