import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__=='__main__':

    # Test arguments
    test_args =  [
        r'D:\Fanuc Experiments\test-master\output\iteration_0',
        r'D:\Fanuc Experiments\test-master\output\iteration_0\iter_mbpo_fig'
    ]

    # Constants
    YLIM = [-0.10, 0.10]
    CONTROL_FILE = 'LineTrackMbpoControl.csv'

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    # Load control
    path = os.path.join(args.data_dir, CONTROL_FILE)
    df = pd.read_csv(path)

    # Plot
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

    plt.show(block=False)
    plt.pause(10)
    sys.exit(0)