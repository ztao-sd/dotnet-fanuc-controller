import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from mbpo import *

if __name__=='__main__':

    # Test arguments
    test_args =  [
        r'D:\Fanuc Experiments\mbpo-6d\test-0708\fast_test\output\iteration_0',
        r'D:\Fanuc Experiments\mbpo-6d\test-0708\fast_test\output\iteration_0\iter_mbpo_fig'
    ]


    # Constants
    YLIM = [-0.10, 0.10]
    CONTROL_FILE = 'LineTrackMbpoControl.csv'
    INPUT_DIM = 12
    OUTPUT_DIM = 6
    OBS_LOW = [-1800, -1200, -100, 2.90, -0.15, 1.4, -0.50, -0.50, -0.50, -0.002, -0.002, -0.002]
    OBS_HIGH = [-1600, -400, 0, 3.20, 0.30, 1.60, 0.50, 0.50, 0.50, 0.002, 0.002, 0.002]
    ACTION_LOW = [-0.08, -0.08, -0.08, -0.003, -0.003, -0.003]
    ACTION_HIGH = [ 0.08, 0.10, 0.10, 0.003, 0.003, 0.003]

    # MBPO 
    observation_space = spaces.Box(low=np.array([-1.0]*INPUT_DIM), high=np.array([1.0]*INPUT_DIM), shape=(INPUT_DIM,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1.0]*OUTPUT_DIM), high=np.array([1.0]*OUTPUT_DIM), shape=(OUTPUT_DIM,), dtype=np.float32)
    mbpo_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'hidden_size': [60, 50],
        'lr': 5e-3,
        'stop_training_threshold': 2e-4,
        'model_buffer_size': 100_000,
        'env_buffer_size': 100_000,
        'gaussian': False,
        'verbose':False
    }
    mbpo = MBPO(**mbpo_args_dict)

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    # Load MBPO control signal vs time
    path = os.path.join(args.data_dir, CONTROL_FILE)
    df = pd.read_csv(path)

    # Plot MBPO control signal vs time
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    t, x, y, z, gamma, beta, alpha = df['time'], df['x'], df['y'], df['z'], df['gamma'], df['beta'], df['alpha']
    ax[0].plot(t, x, label='x')
    ax[0].plot(t, y, label='y')
    ax[0].plot(t, z, label='z')
    ax[0].set_xlabel('time (sec)')
    ax[0].set_title('MBPO Control XYZ')
    ax[0].set_ylim(YLIM)

    ax[1].plot(t, gamma, label='gamma')
    ax[1].plot(t, beta, label='beta')
    ax[1].plot(t, alpha, label='alpha')
    ax[1].set_xlabel('time (sec)')
    ax[1].set_title('MBPO Control WPR')
    ax[1].set_ylim([-0.01, 0.01])
    
    # if legend:
    #     ax.legend()
    if args.save_path is not None:
        fig.savefig(args.save_path)


    # Process and load episode data
    obs_space = spaces.Box(low=np.array(OBS_LOW), high=np.array(OBS_HIGH), shape=(INPUT_DIM,), dtype=np.float32)
    act_space = spaces.Box(low=np.array(ACTION_LOW), high=np.array(ACTION_HIGH), shape=(OUTPUT_DIM,), dtype=np.float32)
    mbpo.env_data_from_files([args.data_dir], obs_space, act_space)
    csv_path = os.path.join(args.data_dir, 'env_buffer.csv')
    mbpo.env_buffer.save_csv(csv_path)
    df = pd.read_csv(csv_path)
    
    # Episodic cumulative rewards
    rewards = df['reward'].tolist()
    cum_reward = sum(rewards[15:])
    
    # Episodic result table
    rows = ['Cumulative Reward']
    data = [[cum_reward]]
    fig, ax = plt.subplots()
    ax.table(cellText=data, rowLabels=rows, loc='best')
    ax.axis('off')
    fig.tight_layout()
    path = os.path.join(args.data_dir, 'result_table')
    fig.savefig(path)

    plt.show(block=False)
    plt.pause(10)
    sys.exit(0)