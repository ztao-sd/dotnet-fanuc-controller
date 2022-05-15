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
        r'D:\Fanuc Experiments\mbpo\test-0503\debug\output\iteration_1',
        r'D:\Fanuc Experiments\mbpo\test-0503\debug\output\iteration_1\iter_mbpo_fig'
    ]


    # Constants
    YLIM = [-0.10, 0.10]
    CONTROL_FILE = 'LineTrackMbpoControl.csv'
    OBS_LOW = [-2000, -1200, -200, -0.50, -0.50, -0.50]
    OBS_HIGH = [-1600, -400, 0, 0.50, 0.50, 0.50]
    ACTION_LOW = [-0.005, -0.005, -0.005]
    ACTION_HIGH = [0.005, 0.005, 0.005]

    # MBPO 
    observation_space = spaces.Box(low=np.array([-1.0]*6), high=np.array([1.0]*6), shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1.0]*3), high=np.array([1.0]*3), shape=(3,), dtype=np.float32)
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
    fig, ax = plt.subplots()
    t, x, y, z = df['time'], df['x'], df['y'], df['z']
    ax.plot(t, x, label='x')
    ax.plot(t, y, label='y')
    ax.plot(t, z, label='z')
    ax.set_xlabel('time (sec)')
    ax.set_title('MBPO Control')
    ax.set_ylim(YLIM)
    # if legend:
    #     ax.legend()
    if args.save_path is not None:
        fig.savefig(args.save_path)

    # Process and load episode data
    obs_space = spaces.Box(low=np.array(OBS_LOW), high=np.array(OBS_HIGH), shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=np.array(ACTION_LOW), high=np.array(ACTION_HIGH), shape=(3,), dtype=np.float32)
    mbpo.env_data_from_files([args.data_dir], obs_space, act_space)
    csv_path = os.path.join(args.data_dir, 'env_buffer.csv')
    mbpo.env_buffer.save_csv(csv_path)
    df = pd.read_csv(csv_path)
    
    # Episodic cumulative rewards
    rewards = df['reward'].tolist()
    cum_reward = sum(rewards[55:])
    
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