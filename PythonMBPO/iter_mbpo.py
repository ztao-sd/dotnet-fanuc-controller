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
    OBS_HIGH = [30, 0.20, 0.20, 0.20]
    ACTION_LOW = [-0.05, -0.05, -0.05]
    ACTION_HIGH = [0.05, 0.05, 0.05]

    # Test arguments
    test_args = [
        '500',
        r'D:\Fanuc Experiments\test-master\output\models',
        r'D:\Fanuc Experiments\test-master\output\rl_policies',
        r'D:\Fanuc Experiments\test-master\output'
    ]

    # Argument parsing
    parser = argparse.ArgumentParser(description='mbpo iteration')
    parser.add_argument('n_epoch', type=int, help='number of epochs')
    parser.add_argument('warmup', type=bool, help='warmup phase?')
    parser.add_argument('model_dir', type=str, help='model save dir')
    parser.add_argument('rl_dir', type=str, help='rl save dir')
    parser.add_argument('data_dirs', type=str, nargs='+', help='list of data dirs')
    args = parser.parse_args(test_args)

    # Environment parameters
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
    gradient_steps = 500  # 500 per episode
    horizon = 1
    # model_gen_data_size = 10_000

    # Load models
    if args.warmup:
        try:
            model_path = os.path.join(args.model_dir, 'best_model.pt')
            mbpo.load_model(model_path)
        except Exception as e:
            print(e)
    else:
        model_path = os.path.join(args.model_dir, 'best_model.pt')
        mbpo.load_model(model_path)
        actor_path = os.path.join(args.rl_dir, 'best_actor.pt')
        td3.load_actor(actor_path)
        critic_path = os.path.join(args.rl_dir, 'best_critic.pt')

    # Load training data into environment buffer
    mbpo.env_data_from_files(args.data_dirs)

    # Train predictive model
    mbpo.load_dataset(mbpo.env_buffer, mbpo.env_buffer.pos, 0.99)
    mbpo.train_model(n_epochs=n_epochs, stop_training_threshold=mbpo.stop_training_threshold, verbose=1)

    # Reinforcement learning
    if not args.warmup:
        model_gen_data_size = 2 * mbpo.env_buffer.pos
        mbpo.generate_data_from_model(n_timesteps=model_gen_data_size, k=horizon, rand=False, agent=td3)
        combined_buffer = ReplayBuffer.combine(mbpo.env_buffer, mbpo.model_buffer)
        td3.train(combined_buffer, gradient_steps)    

    # Evaluate


    # Save models
    actor_path = os.path.join(args.rl_dir, 'best_actor.pt')
    critic_path = os.path.join(args.rl_dir, 'best_critic.pt')
    td3.save_policy(actor_path, critic_path)
    actor_path = os.path.join(args.rl_dir, 'best_actor.onnx')
    td3.save_actor_onnx(actor_path)

    # Save buffers