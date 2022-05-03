from numpy import save
from pnn import *
import argparse
import sys

if __name__ == '__main__':

    # Test arguments
    test_args = [
        '1000',
        r'D:\Fanuc Experiments\test-master\output\pnn_model\mlpg_0416.onnx',
        r'D:\Fanuc Experiments\test-master\output'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='pnn iteration plot')
    parser.add_argument('n_epoch', type=int, help='number of epochs')
    parser.add_argument('onnx_path', type=str, help='onnx save path')
    parser.add_argument('data_dirs', type=str, nargs='+', help='list of data dirs')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    # Load training data
    input_arrays = []
    target_arrays = []
    for data_dir in args.data_dirs:
        control_arrays, error_arrays = PNN.load_data(data_dir)
        input_array, target_array = PNN.shift_data(1, control_arrays, error_arrays)
        input_arrays.append(input_array)
        target_arrays.append(target_array)
    input_array = np.concatenate(input_arrays, axis=0)
    target_array = np.concatenate(target_arrays, axis=0)

    # Train neural network
    pnn_kwargs = {
        'inputs': input_array,
        'targets': target_array,
        'lr': 1e-2,
        'train_split': 0.8,
        'hidden_layers': [100, 100],
        'max_error': 0.30,
        'max_control': 0.10
    }
    pnn = PNN(**pnn_kwargs)
    # Try loading the model dict
    save_dir = os.path.dirname(args.onnx_path)
    model_split = os.path.splitext(args.onnx_path)
    model_path = model_split[0] + '.pt'
    try:
        pnn.load(model_path)
    except Exception as e:
        print(e)
    # Training
    pnn.learn(n_epochs=args.n_epoch)
    pnn.evaluate()

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    pnn.save(model_path)
    pnn.save_onnx(args.onnx_path)

    # Plot learning curves
    fig, ax = plt.subplots()
    x = [i+1 for i in range(len(pnn.losses))]
    ax.plot(x, pnn.losses)
    ax.set_title('Learning Curve')
    save_path = os.path.join(save_dir, 'learning_curve')
    fig.savefig(save_path)

    # Plot validation curve
    fig, ax = plt.subplots()
    x = [i+1 for i in range(len(pnn.eval_losses))]
    ax.plot(x, pnn.eval_losses)
    ax.set_title('Validation Curve')
    save_path = os.path.join(save_dir, 'validation_curve')
    fig.savefig(save_path)

    plt.show(block=False)
    plt.pause(1)
    sys.exit(0)
    



