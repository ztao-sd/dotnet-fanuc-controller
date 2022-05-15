import argparse
import sys
import shutil
from gym import Env
from gym import spaces
from td3 import *
from mbpo import *

class FanucEnvironment(Env):

    pass


if __name__ == '__main__':

    # FILE NAMES
    POSE = 'LineTrackPose.csv'
    ERROR = 'LineTrackError.csv'
    CONTROL = 'LineTrackMbpoControl.csv'
    MODEL = 'best_model.pt'
    ACTOR = 'best_actor.pt'
    CRITIC = 'best_critic.pt'
    ACTOR_ONNX = 'actor.onnx'

    # TD3 HYPERPARAMETERS
    ACTOR_HIDDEN_SIZE = [60, 50]
    CRITIC_HIDDEN_SIZE = [120, 100]
    LEARN_DELAY = 0
    POLICY_DELAY = 2
    TD3_GRAD_STEPS = 1
    TD3_BATCH_SIZE = 100
    GAMMA = 0.98
    ACTOR_LR = 1e-3
    CRITIC_LR = 1e-3
    TAU = 5e-3
    TARGET_POLICY_NOISE = 0.2
    TARGET_POLICY_CLIP = 0.5
    EXPLORATION_NOISE = 0.1

    # MBPO HYPERPARAMETERS
    OBS_LOW = [-2000, -1200, -200, -0.50, -0.50, -0.50]
    OBS_HIGH = [-1600, -400, 0, 0.50, 0.50, 0.50]
    ACTION_LOW = [-0.005, -0.005, -0.005]
    ACTION_HIGH = [0.005, 0.005, 0.005]
    MODEL_HIDDEN_SIZE = [60, 50]
    MODEL_LR = 1e-3
    STOP_TRAINING_THRESHOLD = 2e-4
    MODEL_BUFFER_SIZE = 100_000
    ENV_BUFFER_SIZE = 100_000
    # N_EPOCH
    # MBPO_GRAD_STEPS
    HORIZON = 1
    GEN_DATA_RATIO = 0
    C_REWARD_THRESHOLD = -100_0000

    # C# HYPERPARAMETERS
    """
    // Hyperparameters
    EvalInterval = 5000;
    WarmupIters = 1;
    NEpochs = 700;
    GradientSteps = 1500;
    TrainingIters = 20;
    ExplorationIters = 20;
    ExplorationNoise = new double[3] { 0.003, 0.003, 0.003};
    WarmupNoise = 0.05; // ratio of min/max control
    minControl = new double[3] { -0.02, -0.02, -0.02 };
    maxControl = new double[3] { 0.02, 0.02, 0.02 };
    minState = new double[4] { 0, -0.50, -0.50, -0.50 };
    maxState = new double[4] { 20, 0.50, 0.50, 0.50 };
    """
    
    # Test arguments
    test_args = [
        '20',
        '1',
        '500',
        '0',
        r'D:\Fanuc Experiments\mbpo\test-0514\run-8\output\mbpo',
        r'D:\Fanuc Experiments\mbpo\test-0514\run-8\output\mbpo',
        r'D:\Fanuc Experiments\mbpo\test-0514\run-8\output\iteration_0',
        r'D:\LocalRepos\dotnet-fanuc-controller\PythonMBPO',
        r'D:\Fanuc Experiments\mbpo\test-0514\run-8\output'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='mbpo iteration')
    parser.add_argument('n_epoch', type=int, help='number of epochs')
    parser.add_argument('warmup', type=int, help='warmup phase?')
    parser.add_argument('gradient_steps', type=int, help='gradient steps?')
    parser.add_argument('model_usage', type=int, help='use model?')
    parser.add_argument('model_dir', type=str, help='model save dir')
    parser.add_argument('rl_dir', type=str, help='rl save dir')
    parser.add_argument('iter_dir', type=str, help='iteration dir')
    parser.add_argument('script_dir', type=str, help='script dir')
    parser.add_argument('data_dirs', type=str, nargs='+', help='list of data dirs')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0515\warmup\output')
    args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0515\run-1\output')
    args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0515\run-2\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0514\run-9\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0514\run-2\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0514\run-3\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0512\run-0\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0511\run-1\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0511\run-2\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0510\iterations\run-1\output')
    # args.data_dirs.append(r'D:\Fanuc Experiments\mbpo\test-0510\iterations\run-2\output')


    # RL environment parameters
    observation_space = spaces.Box(low=np.array([-1.0]*6), high=np.array([1.0]*6), shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1.0]*3), high=np.array([1.0]*3), shape=(3,), dtype=np.float32)
    max_action = 1.0
    min_action = -1.0

    # TD3
    td3_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'max_action': max_action,
        'min_action':min_action,
        'hidden_size_actor': ACTOR_HIDDEN_SIZE,
        'hidden_size_critic': CRITIC_HIDDEN_SIZE,
        'learn_delay': LEARN_DELAY,
        'policy_delay': POLICY_DELAY,
        'gradient_steps': TD3_GRAD_STEPS,
        'batch_size': TD3_BATCH_SIZE,
        'gamma': GAMMA,
        'lr_actor': ACTOR_LR,
        'lr_critic': CRITIC_LR,
        'tau': TAU,
        'target_policy_noise': TARGET_POLICY_NOISE,
        'target_noise_clip': TARGET_POLICY_CLIP,
        'exploration_noise': EXPLORATION_NOISE,
        'verbose': 0
    }
    td3 = TD3(**td3_args_dict)

    # MBPO
    mbpo_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'hidden_size': MODEL_HIDDEN_SIZE,
        'lr': MODEL_LR,
        'stop_training_threshold': STOP_TRAINING_THRESHOLD,
        'model_buffer_size': MODEL_BUFFER_SIZE,
        'env_buffer_size': ENV_BUFFER_SIZE,
        'gaussian': False,
        'verbose':False
    }
    mbpo = MBPO(**mbpo_args_dict)

    # More Hyperparameters
    n_epochs = args.n_epoch
    gradient_steps = args.gradient_steps  # ~500 per episode
    horizon = HORIZON
    # model_gen_data_size = 10_000

    # Load models
    if args.warmup:
        try:
            model_path = os.path.join(args.model_dir, MODEL)
            mbpo.load_model(model_path)
        except Exception as e:
            print(e)
    else:
        model_path = os.path.join(args.model_dir, MODEL)
        mbpo.load_model(model_path)
        actor_path = os.path.join(args.rl_dir, ACTOR)
        td3.load_actor(actor_path)
        critic_path = os.path.join(args.rl_dir, CRITIC)

    # Load training data into environment buffer
    obs_space = spaces.Box(low=np.array(OBS_LOW), high=np.array(OBS_HIGH), shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=np.array(ACTION_LOW), high=np.array(ACTION_HIGH), shape=(3,), dtype=np.float32)
    iter_dirs = []
    for dir in args.data_dirs:
        for iter_dir in os.listdir(dir):
            # Check cumulative reward threshold
            if MBPO.iteration_prefix in iter_dir:
                dir_ = os.path.join(dir, iter_dir)
                mbpo.env_data_from_files([dir_], obs_space, act_space)
                csv_path = os.path.join(dir_, 'env_buffer.csv')
                mbpo.env_buffer.save_csv(csv_path)
                mbpo.env_buffer.reset(True)
                df = pd.read_csv(csv_path)
                rewards = df['reward'].tolist()
                cum_reward = sum(rewards[55:])
                if cum_reward > C_REWARD_THRESHOLD:
                    iter_dirs.append(os.path.join(dir_))
        #iter_dirs += [os.path.join(dir, iter_dir) for iter_dir in os.listdir(dir)]
    mbpo.env_data_from_files(iter_dirs, obs_space, act_space)
    env_path = os.path.join(args.model_dir, 'env_buffer.json')
    mbpo.env_buffer.save_buffer(env_path)
    env_path = os.path.join(args.model_dir, 'env_buffer.csv')
    mbpo.env_buffer.save_csv(env_path)

    # Train predictive model
    mbpo.load_dataset(mbpo.env_buffer, mbpo.env_buffer.pos, 0.99)
    mbpo.train_model(n_epochs=n_epochs, stop_training_threshold=mbpo.stop_training_threshold, verbose=1)
    path = os.path.join(args.iter_dir, 'mbpo_learning_curve')
    mbpo.plot_learning_curve(path)

    # Reinforcement learning
    if not args.warmup:
        model_gen_data_size = 0
        if args.model_usage:
            model_gen_data_size = int(GEN_DATA_RATIO * mbpo.env_buffer.pos)
        mbpo.generate_data_from_model(n_timesteps=model_gen_data_size, k=horizon, rand=False, agent=td3, reward_function=mbpo.line_track_reward)
        combined_buffer = ReplayBuffer.combine([mbpo.env_buffer, mbpo.model_buffer])
        td3.train(combined_buffer, gradient_steps, verbose=True)  
        buffer_path = os.path.join(args.iter_dir, 'combined_buffer.csv') 
        combined_buffer.save_csv(buffer_path)

    # Evaluate
    # To be implemented

    # Save models
    model_path = os.path.join(args.rl_dir, MODEL)
    mbpo.save_model(model_path)
    actor_path = os.path.join(args.rl_dir, ACTOR)
    critic_path = os.path.join(args.rl_dir, CRITIC)
    td3.save_policy(actor_path, critic_path)
    actor_path = os.path.join(args.rl_dir, ACTOR_ONNX)
    td3.save_actor_onnx(actor_path)

    # Save models in 'iter_dir'
    model_path = os.path.join(args.iter_dir, MODEL)
    mbpo.save_model(model_path)
    actor_path = os.path.join(args.iter_dir, ACTOR)
    critic_path = os.path.join(args.iter_dir, CRITIC)
    td3.save_policy(actor_path, critic_path)
    actor_path = os.path.join(args.iter_dir, ACTOR_ONNX)
    td3.save_actor_onnx(actor_path)

    # Save buffers
    # To be implemented

    # Copy python scripts to directory of iteration
    src = os.path.join(args.script_dir, 'iter_mbpo.py')
    dst = os.path.join(args.rl_dir, 'iter_mbpo.py')
    shutil.copy2(src, dst)
    src = os.path.join(args.script_dir, 'iter_mbpo_plot.py')
    dst = os.path.join(args.rl_dir, 'iter_mbpo_plot.py')
    shutil.copy2(src, dst)

    plt.show(block=False)
    plt.pause(1)
    sys.exit(0)