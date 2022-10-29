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

    pid_ilc_prev_control_file_name = 'LineTrackPidIlcPrevControl.csv'
    pid_ilc_control_file_name = 'LineTrackPidIlcControl.csv'
    pid_ilc_error_file_name = 'LineTrackPidIlcError.csv'
    pid_ilc_error_dot_file_name = 'LineTrackPidIlcErrorDot.csv'

    def iter_plot(data_dir, save_path=None):

        # Read data
        error_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.path_error_file_name))
        control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.control_file_name))
        #ilc_control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.ilc_control_file_name

        # Define subplot
        def iter_subplot(pd, ax, subtitle, xlabel='time (sec)', ylim=[-0.3, 0.3], legend=False):
            t, x, y, z = pd['time'], pd['x'], pd['y'], pd['z']
            array = pd.iloc[:, 1:4].to_numpy().reshape(-1, 3)
            dist = np.sqrt(np.sum(array**2, axis=1)).reshape(-1, 1)
            ax.plot(t,x, label='x')
            ax.plot(t,y, label='y')
            ax.plot(t,z, label='z')
            ax.plot(t,dist, label='dist')
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle)
            ax.set_ylim(ylim)
            if legend:
                ax.legend()
        
        def iter_subplot_orientation(pd, ax, subtitle, xlabel='time (sec)', ylim=[-0.20, 0.20], legend=False):
            t, x, y, z = pd['time'], pd['gamma'], pd['beta'], pd['alpha']
            ax.plot(t,x*180/np.pi, label='gamma')
            ax.plot(t,y*180/np.pi, label='beta')
            ax.plot(t,z*180/np.pi, label='alpha')
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
        iter_subplot_orientation(error_pd, ax[1,0], 'Orientation Error (deg)')
        # Plot control signal WPR
        iter_subplot_orientation(control_pd, ax[1,1], 'Orientation (deg)', legend=True, ylim=[-0.03, 0.03])

        

        fig.suptitle('Iteration Plot')
        fig.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path)

    def iter_ilc_plot(data_dir, save_path_xyz=None, save_path_wpr=None):

        # Read data
        ilc_error_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.pid_ilc_error_file_name))
        ilc_error_dot_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.pid_ilc_error_dot_file_name))
        ilc_control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.pid_ilc_control_file_name))
        ilc_prev_control_pd = pd.read_csv(os.path.join(data_dir, IlcPlotting.pid_ilc_prev_control_file_name))

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
            ax.plot(t,x*180/np.pi, label='gamma')
            ax.plot(t,y*180/np.pi, label='beta')
            ax.plot(t,z*180/np.pi, label='alpha')
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle)
            ax.set_ylim(ylim)
            if legend:
                ax.legend()

        # Plot XYZ
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        iter_subplot(ilc_prev_control_pd, ax[0,0], "Prev. Control (mm)", ylim=[-0.05, 0.05])
        iter_subplot(ilc_control_pd, ax[0,1], "Control (mm)", ylim=[-0.05, 0.05])
        iter_subplot(ilc_error_pd, ax[1,0], "Error (mm)", ylim=[-0.20, 0.20], legend=True)
        iter_subplot(ilc_error_dot_pd, ax[1,1], "ErrorDot (mm)", ylim=[-0.03, 0.03])
        fig.suptitle('Iteration ILC Position Plot')
        fig.tight_layout()
        if save_path_xyz is not None:
            fig.savefig(save_path_xyz)

        # Plot WPR
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        iter_subplot_orientation(ilc_prev_control_pd, ax[0,0], "Prev. Control (mm)", ylim=[-0.05, 0.05])
        iter_subplot_orientation(ilc_control_pd, ax[0,1], "Control (mm)", ylim=[-0.05, 0.05])
        iter_subplot_orientation(ilc_error_pd, ax[1,0], "Error (mm)", ylim=[-0.05, 0.05], legend=True)
        iter_subplot_orientation(ilc_error_pd, ax[1,1], "Error (mm)", ylim=[-0.05, 0.05])
        fig.suptitle('Iteration ILC Orientation Plot')
        fig.tight_layout()
        if save_path_wpr is not None:
            fig.savefig(save_path_wpr)



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