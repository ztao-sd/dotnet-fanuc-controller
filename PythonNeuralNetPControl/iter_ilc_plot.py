from ilc_utility import *

if __name__ == '__main__':

    TEST = True
    # TEST = False

    # Test arguments
    test_args = [
        r'C:\Users\Tao\LocalRepos\Fanuc Experiments\mbpo\test-0516\eval-2\output\iteration_3',
        r'C:\Users\Tao\LocalRepos\Fanuc Experiments\mbpo\test-0516\eval-2\output\iteration_3\iter_ilc_fig_xyz',
        r'C:\Users\Tao\LocalRepos\Fanuc Experiments\mbpo\test-0516\eval-2\output\iteration_3\iter_ilc_fig_wpr'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path_xyz', type=str, nargs='?', help='figure save path xyz')
    parser.add_argument('save_path_wpr', type=str, nargs='?', help='figure save path wpr')
    if TEST:
        args = parser.parse_args(test_args)
    else:
        args = parser.parse_args()
    

    IlcPlotting.iter_ilc_plot(args.data_dir, args.save_path_xyz, args.save_path_wpr)
    plt.show(block=False)
    plt.pause(10)
    sys.exit(0)