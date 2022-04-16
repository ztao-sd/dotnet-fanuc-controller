from utils import *
from td3 import *
from callback import *
from mbpo import *
from gym import spaces

if __name__ == '__main__':
    
    # Bounds
    OBS_LOW = -0.20
    OBS_HIGH = 0.20
    ACTION_LOW = -0.10
    ACTION_HIGH = 0.10

    #region TD3
    td3_args_dict={
        'observation_space': spaces.Box(low=np.array([0, -1.0, -1.0, -1.0]), high=np.array([1.0]*4), shape=(4,), dtype=np.float32),
        'action_space': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        'max_action': 1.0,
        'min_action':-1.0,
        'hidden_size_actor': [60, 60],
        'hidden_size_critic': [100, 100],
        'learn_delay': 0,
        'policy_delay': 2,
        'gradient_steps': 1,
        'batch_size': 100,
        'gamma': 0.98,
        'lr_actor': 1e-3,
        'lr_critic': 1e-3,
        'tau': 5e-3,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'exploration_noise': 0.1,
        'verbose': 0
    }
    td3 = TD3(**td3_args_dict)
    #endregion

    #region MBPO
    mbpo_args_dict={
        'observation_space' = td3.observation_space,
        'action_space' = td3.action_space,
        # Model
        'hidden_size' = [100, 100],
        'lr' = 1e-3,
        stop_training_threshold,
    }
    #endregion
