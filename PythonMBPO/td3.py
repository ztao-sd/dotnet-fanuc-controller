import copy
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_size[0])
		self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
		self.l3 = nn.Linear(hidden_size[1], action_dim)
		
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * T.tanh(self.l3(a))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
		self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
		self.l3 = nn.Linear(hidden_size[1], 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_size[0])
		self.l5 = nn.Linear(hidden_size[0],hidden_size[1])
		self.l6 = nn.Linear(hidden_size[1], 1)


	def forward(self, state, action):
		sa = T.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1_forward(self, state, action):
		sa = T.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class TD3():

    def __init__(
        self,
        # Environment
        observation_space,
        action_space,
        max_action,
        min_action,
        # Model
        hidden_size_actor,
        hidden_size_critic,
        # Training hypers
        learn_delay,
        policy_delay,
        gradient_steps,
        batch_size,
        gamma,
        lr_actor,
        lr_critic,
        tau,
        target_policy_noise,
        target_noise_clip,
        exploration_noise,
        verbose=0
    ):
        #Device
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
        # Environment
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_action = max_action
        self.min_action = min_action
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        # Model
        self.hidden_size_actor = hidden_size_actor
        self.hidden_size_critic = hidden_size_critic

        # Trainin hypers
        self.learn_delay = learn_delay
        self.policy_delay = policy_delay
        self.gradients_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.verbose = verbose

        # Model
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size_actor, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_size_actor, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size_critic).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size_critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training tracking
        self.train_step = 0
        self.time_step = 0
    
    def select_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.device)
        return self.actor(state).cpu().data.numpy()
    
    def exploration_action(self, state):
        action = self.select_action(state)
        noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action.astype(np.float32)

    def soft_update(self, local_model, target_model, tau=None):
        # Soft update value
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # target_param.data.mul_(1 - tau)
            # T.add(target_param.data, local_param.data, alpha=tau, out=target_param.data)
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, buffer, gradient_steps=None, verbose=False):

        if self.time_step < self.learn_delay:
            return
        # Train mode
        self.actor.train(True)
        self.critic.train(True)

        if gradient_steps is not None:
            self.gradient_steps = gradient_steps

        for idx in range(self.gradient_steps):

            self.train_step += 1
            # Sample buffer
            replay_data = buffer.sample(self.batch_size)
            observations = replay_data[0]
            actions = replay_data[1]
            rewards = replay_data[2]
            next_observations = replay_data[3]
            dones = replay_data[4]

            with T.no_grad():
                noise = (T.rand_like(
                    actions)*self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = self.actor_target(next_observations)
                next_actions = next_actions + noise
                next_actions = T.clamp(next_actions, -self.max_action, self.max_action)

                next_q_values1, next_q_values2 = self.critic_target(next_observations, next_actions)
                next_q_values = T.minimum(next_q_values1, next_q_values2)
                target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values
            
            current_q_values = self.critic(observations, actions)
            critic_loss = F.mse_loss(current_q_values[0],target_q_values) + F.mse_loss(current_q_values[1],target_q_values)
            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # Delayed policy update
            if self.train_step % self.policy_delay == 0:
                # Actor loss
                actor_loss = -T.mean(self.critic.q1_forward(observations, self.actor(observations)))
                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # Soft update target network
                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.critic, self.critic_target)

                # Print
                if verbose:
                    print(f'Gradient Steps: {idx} | Critic Loss: {critic_loss.item()} | Actor Loss: {actor_loss.item()}')

            
        self.actor.train(False)
        self.critic.train(False)

    def load_critic(self, critic_path):
        self.critic.load_state_dict(T.load(critic_path))
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def load_actor(self, actor_path):
        self.actor.load_state_dict(T.load(actor_path))
        self.actor_target.load_state_dict(self.actor.state_dict())

    def save_policy(self, actor_path, critic_path):
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)
    
    def save_actor_onnx(self, actor_path):
        dummy_input = T.randn(self.state_dim, device=self.device)
        input_names = ['state']
        output_names = ['control']
        T.onnx.export(self.actor, dummy_input, actor_path, input_names=input_names,
                        output_names=output_names, verbose=True)