import argparse
from gym import Env
from gym import spaces
from td3 import *
from mbpo import *

class FanucEnvironment(Env):

    pass


if __name__ == '__main__':

    # Constants
    ERROR = 'LineTrackError.csv'
    CONTROL = 'LineTrackMbpoControl.csv'
    MODEL = 'best_model.pt'
    ACTOR = 'best_actor.pt'
    CRITIC = 'best_critic.pt'
    ACTOR_ONNX = 'actor.onnx'
    OBS_LOW = [0, -0.20, -0.20, -0.20]
    OBS_HIGH = [20, 0.20, 0.20, 0.20]
    ACTION_LOW = [-0.05, -0.05, -0.05]
    ACTION_HIGH = [0.05, 0.05, 0.05]
    ACTION_LOW = [-0.20] * 3
    ACTION_HIGH = [0.20] * 3

    # Test arguments
    test_args = [
        '20',
        '0',
        '500',
        r'D:\Fanuc Experiments\test-master\output\mbpo',
        r'D:\Fanuc Experiments\test-master\output\mbpo',
        r'D:\Fanuc Experiments\test-master\output'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='mbpo iteration')
    parser.add_argument('n_epoch', type=int, help='number of epochs')
    parser.add_argument('warmup', type=int, help='warmup phase?')
    parser.add_argument('gradient_steps', type=int, help='gradient steps?')
    parser.add_argument('model_dir', type=str, help='model save dir')
    parser.add_argument('rl_dir', type=str, help='rl save dir')
    parser.add_argument('data_dirs', type=str, nargs='+', help='list of data dirs')
    args = parser.parse_args()
    # args = parser.parse_args(test_args)

    # RL environment parameters
    observation_space = spaces.Box(low=np.array([-1.0]*4), high=np.array([1.0]*4), shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1.0]*3), high=np.array([1.0]*3), shape=(3,), dtype=np.float32)
    max_action = 1.0
    min_action = -1.0

    # TD3
    td3_args_dict={
        'observation_space': observation_space,
        'action_space': action_space,
        'max_action': max_action,
        'min_action':min_action,
        'hidden_size_actor': [60, 50],
        'hidden_size_critic': [120, 100],
        'learn_delay': 0,
        'policy_delay': 2,
        'gradient_steps': 1,
        'batch_size': 100,
        'gamma': 0.99,
        'lr_actor': 1e-3,
        'lr_critic': 1e-3,
        'tau': 5e-3,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'exploration_noise': 0.1,
        'verbose': 0
    }
    td3 = TD3(**td3_args_dict)

    # MBPO
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

    # More Hyperparameters
    n_epochs = args.n_epoch
    gradient_steps = args.gradient_steps  # ~500 per episode
    horizon = 1
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
    obs_space = spaces.Box(low=np.array(OBS_LOW), high=np.array(OBS_HIGH), shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=np.array(ACTION_LOW), high=np.array(ACTION_HIGH), shape=(3,), dtype=np.float32)
    iter_dirs = []
    for dir in args.data_dirs:
        iter_dirs += [os.path.join(dir, iter_dir) for iter_dir in os.listdir(dir)]
    mbpo.env_data_from_files(iter_dirs, obs_space, act_space)
    env_path = os.path.join(args.model_dir, 'env_buffer.json')
    mbpo.env_buffer.save_buffer(env_path)
    env_path = os.path.join(args.model_dir, 'env_buffer.csv')
    mbpo.env_buffer.save_csv(env_path)

    # Train predictive model
    mbpo.load_dataset(mbpo.env_buffer, mbpo.env_buffer.pos, 0.99)
    mbpo.train_model(n_epochs=n_epochs, stop_training_threshold=mbpo.stop_training_threshold, verbose=1)

    # Reinforcement learning
    if not args.warmup:
        model_gen_data_size = 2 * mbpo.env_buffer.pos
        mbpo.generate_data_from_model(n_timesteps=model_gen_data_size, k=horizon, rand=False, agent=td3, reward_function=mbpo.line_track_reward)
        combined_buffer = ReplayBuffer.combine([mbpo.env_buffer, mbpo.model_buffer])
        td3.train(combined_buffer, gradient_steps, verbose=True)    

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

    # Save buffers
    # To be implemented