import gym
import numpy as np
import torch as T
import onnx
import onnxruntime as ort
import json
import os


class ReplayBuffer():

    def __init__(self, observation_space, action_space, buffer_size=1_000_000) -> None:
        self.buffer_size = buffer_size
        self.observations_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.pos = 0
        self.full = False
        # Init buffer
        self.observations = np.zeros(
            (self.buffer_size,) + self.observations_shape, dtype=np.float32)
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.buffer_size,) + self.observations_shape, dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def reset(self):
        self.pos = 0
        self.full = False

    def to_torch(self, array, copy=True):
        if copy:
            return T.tensor(array).to(self.device)
        return T.as_tensor(array).to(self.device)

    def add(self, obs, action, rewards, next_obs, done):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(rewards).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        if self.full:
            idx = (np.random.randint(1, self.buffer_size,
                   size=batch_size) + self.pos) % self.buffer_size
        else:
            idx = np.random.randint(0, self.pos, size=batch_size)

        data = (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx]
        )

        return tuple(map(self.to_torch, data))

    @staticmethod
    def combine(self, buffers):
        observation_space = buffers[0].observation_space
        action_space = buffers[0].action_space
        size = sum([buffer.buffer_size for buffer in buffers])
        combined_buffer = ReplayBuffer(observation_space, action_space, size)
        for buffer in buffers:
            if buffer.full:
                n = buffer.buffer_size
            else:
                n = buffer.pos
            for i in range(n):
                obs = buffer.observations[i]
                action = buffer.actions[i]
                reward = buffer.rewards[i]
                next_obs = buffer.next_observations[i]
                done = buffer.dones[i]
                combined_buffer.add(obs, action, reward, next_obs, done)
        return combined_buffer

    def _list_to_json(self):
        dict = {
            "full": self.full,
            "pos": self.pos,
            "buffer": [
                {
                    "observations": observations.tolist(),
                    "action": action.tolist(),
                    "reward": reward.tolist(),
                    "next_observations": next_observations.tolist(),
                    "done": done.tolist()
                }
                for (observations, action, reward, next_observations, done) in zip(self.observations, self.actions, self.rewards, self.next_observations, self.dones)
            ]
        }
        return dict

    def _json_to_numpy(self, dict):
        self.pos = dict.get("pos")
        self.full = dict.get("full")
        for idx, transition in enumerate(dict.get("buffer")):
            self.observations[idx] = np.array(transition.get("observations"))
            self.actions[idx] = np.array(transition.get("action"))
            self.rewards[idx] = np.array(transition.get("reward"))
            self.next_observations[idx] = np.array(
                transition.get("next_observations"))

    def save_buffer(self, file_path):
        buffer_dict = self._list_to_json()
        with open(file_path, 'w') as f:
            json.dump(buffer_dict, f)

    def load_buffer(self, file_path):
        with open(file_path, 'r') as f:
            buffer_dict = json.load(f)
        self._json_to_numpy(buffer_dict)


class OnnxUtils:

    def __init__(self) -> None:
        pass

    @staticmethod
    def onnx_save_actor(save_path, model, input_shape, device):
        dummy_input = T.randn(input_shape, device=device)
        input_names = ["state"]
        output_names = ["action"]
        T.onnx.export(model, dummy_input, save_path, verbose=True,
                      input_names=input_names, output_names=output_names)
    
    @staticmethod
    def onnx_check(save_path):
        model = onnx.load(save_path)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))


    
if __name__=="__main__":
    from td3 import *
    from mbpo import * 
    import time

    hidden_size_model = [100, 80]
    hidden_size_actor = [60, 50]
    hidden_size_critic = [120, 100]
    prev_top_dir_name = f"220205_001_mbpo{hidden_size_model}_td3{hidden_size_actor}-{hidden_size_critic}_pendulumv0"
    prev_top_dir = os.getcwd()
    prev_experiment_dir = os.path.join(prev_top_dir, "results", prev_top_dir_name)
    prev_rl_policy_save_dir = os.path.join(prev_top_dir, "results", prev_top_dir_name, "rl_policies")
    prev_model_save_dir = os.path.join(prev_top_dir, "results", prev_top_dir_name, "models")
    prev_data_save_dir = os.path.join(prev_top_dir, "results", prev_top_dir_name, "data")

    prev_rl_actor_path = os.path.join(prev_rl_policy_save_dir, "best_actor_arch_60-50.pt")
    prev_rl_critic_path = os.path.join(prev_rl_policy_save_dir, "best_critic_arch_120-100.pt")
    prev_model_path = os.path.join(prev_model_save_dir, "model_eps20000.pt")

    # Environment
    env = gym.make("Pendulum-v0")
    observation_space = env.observation_space
    action_space = env.action_space
    observation_shape = observation_space.shape
    state_dim = observation_space.sample().flatten().size
    action_dim = action_space.sample().flatten().size
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    # Load Pytorch model
    actor = Actor(state_dim, action_dim, hidden_size_actor, max_action)
    actor.load_state_dict(T.load(prev_rl_actor_path))
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # Save onnx model
    top_dir = os.getcwd()
    onnx_dir = os.path.join(prev_experiment_dir, "onnx_models")
    onnx_path = os.path.join(onnx_dir, f"best_actor_arch_{hidden_size_actor}.onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    OnnxUtils.onnx_save_actor(onnx_path, actor, input_shape=state_dim, device=device)
    
    # Check onnx model
    OnnxUtils.onnx_check(onnx_path)

    # Inference session
    ort_session = ort.InferenceSession(onnx_path)
    state = observation_space.sample()
    start_time = time.time()
    action = ort_session.run(None, {"state": state})
    stop_time = time.time()
    print(stop_time-start_time)
    print(state)
    print(action)
