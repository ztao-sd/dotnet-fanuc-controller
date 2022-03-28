from ilc_utility import *

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='iteration plot')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('save_path', type=str, nargs='?', help='figure save path')
    args = parser.parse_args()

    IlcPlotting.iter_ilc_plot(args.data_dir, args.save_path)
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    sys.exit(0)