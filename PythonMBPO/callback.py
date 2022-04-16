from pyexpat import model
import numpy as np
import torch as T
import os

class CallBack():

    def __init__(self, verbose=0) -> None:
        self.n_calls = 0
        self.time_step = 0
        self.verbose = verbose

    def _init_callback(self):
        pass

    def init_callback(self, env, agent, mbpo=None):
        self.env = env
        self.agent = agent
        self.mbpo = mbpo
        self._init_callback()

    def _on_step(self):
        return True

    def on_step(self):
        self.n_calls += 1
        self.time_step = self.agent.time_step
        return self._on_step()

class CheckPointCallback(CallBack):

    def __init__(self, save_freq, model_save_dir=None, data_save_dir=None, verbose=0) -> None:
        super(CheckPointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_save_dir = model_save_dir
        self.data_save_dir = data_save_dir

    def _init_callback(self):
        if self.model_save_dir is not None:
            os.makedirs(self.model_save_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save model
            actor_save_path = os.path.join(self.model_save_dir, f"actor_eps{self.time_step}.pt")
            critic_save_path = os.path.join(self.model_save_dir, f"critic_eps{self.time_step}.pt")
            model_save_path = os.path.join(self.model_save_dir, f"model_eps{self.time_step}.pt")
            self.agent.save_policy(actor_save_path, critic_save_path)
            self.mbpo.save_model(model_save_path)
            model_save_path = os.path.join(self.model_save_dir, f'model_best.pt')
            self.mbpo.save_model(model_save_path)
            # Save training data
            model_buffer_save_path = os.path.join(self.data_save_dir, f"model_buffer_eps{self.time_step}.json")
            env_buffer_save_path = os.path.join(self.data_save_dir, f"env_buffer_eps{self.time_step}.json")
            self.mbpo.model_buffer.save_buffer(model_buffer_save_path)
            self.mbpo.env_buffer.save_buffer(env_buffer_save_path)
            if self.verbose > 0:
                print(f"Saving models checkpoint to {self.model_save_dir}")
        return True

class EvalCallBack(CallBack):
    
    def __init__(
        self, 
        eval_env, 
        n_eval_episodes=5,
        eval_freq=10_000,
        log_path=None,
        best_actor_save_path=None,
        best_critic_save_path=None,
        verbose=0
        ) -> None:
        super(EvalCallBack, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_actor_save_path = best_actor_save_path
        self.best_critic_save_path = best_critic_save_path
        
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_lengths = []
    
    def _init_callback(self):
        
        # Make folders if they don't exist
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.best_actor_save_path is not None:
            os.makedirs(self.best_actor_save_path, exist_ok=True)
        if self.best_critic_save_path is not None:
            os.makedirs(self.best_critic_save_path, exist_ok=True)

    def _evaluate_policy(
        self,
        agent,
        eval_env,
        n_eval_episodes,
        ):

        episode_rewards = []
        episode_lengths = []

        for _ in range(n_eval_episodes):
            observation = eval_env.reset()
            score = 0
            done = False
            length = 0
            while not done:
                # Agent action
                action = agent.select_action(observation)
                observation_, reward, done, _ = eval_env.step(action)
                observation = observation_.copy()
                score += reward
                length += 1

            episode_rewards.append(score)
            episode_lengths.append(length)
        
        return episode_rewards, episode_lengths

    def _on_step(self):
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            episode_rewards, episode_lengths = self._evaluate_policy(
                self.agent,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
            )
            # Compute evaluation metrics
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            # Print info
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.time_step}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Update (and save) best 
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.best_mean_reward = mean_reward
                # Write to txt file
                if self.log_path is not None:
                    try:
                        path = os.path.join(self.log_path, "policy_eval_log")
                        with open(path, 'a') as f:
                            f.write(f"Eval num_timesteps={self.time_step}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}" +"\n")
                    except Exception as e:
                        print(e)
                # Save best model
                if self.best_actor_save_path is not None and self.best_critic_save_path is not None:
                    # self.agent.save_models(self.best_model_save_path)
                    self.agent.save_policy(self.best_actor_save_path, self.best_critic_save_path)
                    if self.verbose > 0:
                        print("Saved best model!")
            return True
        