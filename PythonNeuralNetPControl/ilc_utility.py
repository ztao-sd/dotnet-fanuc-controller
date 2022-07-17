import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
import argparse
import time

class Test:

    def test():
        print('Hello')

class IlcPlotting:

    pose_file_name = 'LineTrackPose.csv'
    path_error_file_name = 'LineTrackError.csv'
    control_file_name = 'LineTrackControl.csv'
    ilc_control_file_name = 'LineTrackIlcControl.csv'

    def iter_plot(data_dir, save_path=None):

        # Read data
        error_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.path_error_file_name))
        control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.control_file_name))
        #ilc_control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.ilc_control_file_name

        # Define subplot
        def iter_subplot(pd, ax, subtitle, xlabel='time (sec)', ylim=[-0.30, 0.30], legend=False):
            t, x, y, z = pd['time'], pd['x'], pd['y'], pd['z']
            ax.plot(t,x, label='x')
            ax.plot(t,y, label='y')
            ax.plot(t,z, label='z')
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle)
            ax.set_ylim(ylim)
            if legend:
                ax.legend()
        
        def iter_subplot_orientation(pd, ax, subtitle, xlabel='time (sec)', ylim=[-0.003, 0.003], legend=False):
            t, x, y, z = pd['time'], pd['gamma'], pd['beta'], pd['alpha']
            ax.plot(t,x, label='gamma')
            ax.plot(t,y, label='beta')
            ax.plot(t,z, label='alpha')
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle)
            ax.set_ylim(ylim)
            if legend:
                ax.legend()

        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        # Plot path error XYZ
        iter_subplot(error_pd, ax[0,0], 'Path Error (mm)')

        # Plot control signal XYZ
        iter_subplot(control_pd, ax[0,1], 'Control Signal (mm)', legend=True)

        # Plot orientation error WPR
        iter_subplot_orientation(error_pd, ax[1,0], 'Orientation Error (rad)')
        # Plot control signal WPR
        iter_subplot_orientation(control_pd, ax[1,1], 'Orientation (deg)', legend=True, ylim=[-0.03, 0.03])

        

        fig.suptitle('Iteration Plot')
        fig.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path)

    def iter_ilc_plot(data_dir, save_path=None):

        # Read data
        ilc_control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.ilc_control_file_name))

        # Define subplot
        def iter_subplot(pd, ax, subtitle, xlabel='time (sec)', ylim=[-0.30, 0.30], legend=False):
            t, x, y, z = pd['time'], pd['x'], pd['y'], pd['z']
            ax.plot(t,x, label='x')
            ax.plot(t,y, label='y')
            ax.plot(t,z, label='z')
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle)
            ax.set_ylim(ylim)
            if legend:
                ax.legend()
        
        fig, ax = plt.subplots(figsize=(5, 5))
        # Plot control signal
        iter_subplot(ilc_control_pd, ax, 'ILC Control Signal (mm)', ylim=[-0.05, 0.05],legend=True)

        fig.suptitle('Iteration ILC Plot')
        fig.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path)

#if __name__=='__main__':

#    parser = argparse.ArgumentParser(description='iteration plot')
#    parser.add_argument('data_dir', type=str, help='data directory')
#    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
#    #args = parser.parse_args()

#    data_dir = r'D:\fanuc experiments\test-0327-pcontrol\output\iteration_1'
#    save_path = r'D:\fanuc experiments\test-0327-pcontrol\output\iteration_1\test_fig'
#    IlcPlotting.iter_plot(data_dir, save_path)

#    #data_dir = r'd:\fanuc experiments\test-0327-pcontrol\output\iteration_1'
#    #save_path = r'd:\fanuc experiments\test-0327-pcontrol\output\iteration_1\test_fig_ilc'
#    #ilcplotting.iter_ilc_plot(data_dir, save_path)

#    plt.show()
#    #plt.show(block=False)
#    #plt.pause(10)
#    #sys.exit(0)