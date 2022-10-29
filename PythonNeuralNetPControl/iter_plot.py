from ilc_utility import *

if __name__ == '__main__':

    # Test arguments
    test_args = [
        r'D:\Fanuc Experiments\test\test-1029\eval-2\output_10-29-2022 18.16.42\iteration_0',
        r'D:\Fanuc Experiments\test\test-1029\eval-2\output_10-29-2022 18.16.42\iteration_0\iter_fig_norm'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
    # args = parser.parse_args()
    args = parser.parse_args(test_args)

    IlcPlotting.iter_plot(args.data_dir, args.save_path)
    plt.show(block=False)
    plt.pause(1000)
    sys.exit(0)