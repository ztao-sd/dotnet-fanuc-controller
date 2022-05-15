import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os
from pnn import *

if __name__ == '__main__':

    # Test arguments
    test_args =  [
        r'D:\Fanuc Experiments\pnn\test-0515\training\output\iteration_0',
        r'D:\Fanuc Experiments\pnn\test-0515\training\output\iteration_0\iter_pnn_fig'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    # Load control
    path = os.path.join(args.data_dir, 'LineTrackPnnControl.csv')
    df = pd.read_csv(path)

    # Plot
    fig, ax = plt.subplots()
    PNNPlot.pose_data_plot(ax, df, 'PNN Control')
    if args.save_path is not None:
        fig.savefig(args.save_path)

    # Plot error vs error
    path = os.path.join(args.data_dir, 'LineTrackError.csv')
    df = pd.read_csv(path)
    path = os.path.join(args.data_dir, 'LineTrackPnnError.csv')
    df_pnn = pd.read_csv(path)
    fig, ax = plt.subplots()
    ax.plot(df.iloc[:,1:4].values)
    ax.plot(df_pnn.iloc[:,1:4].values)
    save_path = os.path.join(args.data_dir, 'error_comparison')
    fig.savefig(save_path)


    plt.show(block=False)
    plt.pause(10)
    sys.exit(0)