import gym
import torch as T
import numpy as np

from td3 import *
from utils import ReplayBuffer
from callback import EvalCallBack

if __name__=="__main__":
    # Create environment
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')
    observation_space = env.observation_space
    action_space = env.action_space
    observation_shape = observation_space.shape
    action_dim = action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    # Create buffer
    buffer_size = 200_000
    buffer = ReplayBuffer(observation_space, action_space, buffer_size)
    # Create model
    hidden_size = [400, 300]
    # Create TD3 agent
    agent = TD3(
        # Environment
        observation_space=observation_space,
        action_space=action_space,
        max_action=max_action,
        min_action=min_action,
        # Model
        hidden_size=hidden_size,
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
        eval_freq=1_000, # timesteps
        log_path=None,
        best_model_save_path=None, 
        verbose=1
    )
    eval_callback.init_callback(env, agent)
    # Learn
    total_timesteps = 20_000
    all_scores = []

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    observation = env.reset()
    score = 0
    done = False
    length = 0
    for t in range (total_timesteps):  
        agent.time_step += 1
        episode_timesteps += 1
        # Select action
        if t < agent.learn_delay:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation)
            noise = np.random.normal(0, agent.exploration_noise, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)
        
        # Perform action
        next_observation, reward, done, _ = env.step(action)

        # Store transition
        buffer.add(observation, action, reward, next_observation, done)
        observation = next_observation
        episode_reward += reward

        # Train
        # if t > agent.learn_delay:
        #     agent.train(buffer, 1)

        # Handle done
        if done:
            agent.train(buffer, episode_timesteps)
            # print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            observation, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluation
        eval_callback.on_step()
