from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import pickle
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from tqdm import tqdm
import time

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)



# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, device='cpu') -> None:
        self.path = 'model.pth'

        config = {'nb_actions': env.action_space.n,
                'learning_rate': 5e-3,
                'gamma': 0.5,
                'buffer_size': 1000000,
                'epsilon_min': 0.01,
                'epsilon_max': 1.,
                'epsilon_decay_period': 10000,
                'epsilon_delay_decay': 500,
                'batch_size': 50,
                'gradient_steps': 1,
                'update_target_strategy': 'replace',
                'update_target_freq': 10,
                'update_target_tau': 0.05,
                'criterion': torch.nn.SmoothL1Loss(),
                'monitoring_nb_trials': 0}

        print(config)

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        nb_neurons=32
        model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                nn.ReLU(),
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, n_action)).to(device)
        


        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=5000, gamma=0.8)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self) -> None:
        self.model = torch.load(self.path, map_location=torch.device('cpu'))

    def collect_samples(self, env=env, horizon=int(6e4)):
        s, _ = env.reset()

        S, A, R, S2, D = [], [], [], [], []

        for _ in tqdm(range(horizon)):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)

            for (L, e) in zip([S, A, R, S2, D], [s, a, r, s2, done]):
                L.append(e)

            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2

        S, A, R, S2, D = np.array(S), np.array(A).reshape((-1,1)), np.array(R), np.array(S2), np.array(D)

        return S, A, R, S2, D

    def rf_fqi(self, S, A, R, S2, D, iterations=400, nb_actions=env.action_space.n, gamma=1):
        nb_samples = S.shape[0]
        fitted_Q = None
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations)):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = fitted_Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            # Q = RandomForestRegressor()
            Q = ExtraTreesRegressor(n_estimators=50, min_samples_split=2, max_features=8)
            Q.fit(SA,value)
            fitted_Q = Q
        return fitted_Q
    
    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.act(x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        elapsed = time.time()
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # action = greedy_action(self.model, state)
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            if len(self.memory) > self.batch_size:
                self.scheduler.step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                print(f"Time elapsed: {time.time() - elapsed:.3f}s")
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.2g}'.format(episode_cum_reward), 
                          ", MC tot ", '{:3.2g}'.format(MC_tr),
                          ", MC disc ", '{:3.2g}'.format(MC_dr),
                          ", V0 ", '{:6.2g}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.2g}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state


if __name__ == '__main__':
    # print("Collecting samples...")
    # samples = agent.collect_samples(training_env, horizon=int(1e4))
    # print("Fitting Q...")
    # agent.learned_Q = agent.rf_fqi(*samples, iterations=400)
    # print("Saving model to disk...")
    # agent.save(agent.path)
    # print("Done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ProjectAgent(device=device)
    print(f"Train agent on {device}")
    ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 200)
    print("Saving model")
    agent.save(agent.path)
    print("Done")




################################################################################################
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class policyNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         state_dim = env.observation_space.shape[0]
#         n_action = env.action_space.n
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, n_action)

#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(dim=0)
#         x = F.relu(self.fc1(x))
#         action_scores = self.fc2(x)
#         return F.softmax(action_scores,dim=1)

#     def sample_action(self, x):
#         probabilities = self.forward(x)
#         action_distribution = Categorical(probabilities)
#         return action_distribution.sample().item()

#     def log_prob(self, x, a):
#         probabilities = self.forward(x)
#         action_distribution = Categorical(probabilities)
#         return action_distribution.log_prob(a)
    
#################################################################################################
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import trange

# class reinforce_agent:
#     def __init__(self, config, policy_network):
#         self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
#         self.scalar_dtype = next(policy_network.parameters()).dtype
#         self.policy = policy_network
#         self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
#         lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
#         self.optimizer = torch.optim.Adam(list(self.policy.parameters()),lr=lr)
#         self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1

#     def sample_action_and_log_prob(self, x):
#         probabilities = self.policy(torch.as_tensor(x))
#         action_distribution = Categorical(probabilities)
#         action = action_distribution.sample()
#         log_prob = action_distribution.log_prob(action)
#         return action.item(), log_prob
    
#     def one_gradient_step(self, env):
#         # run trajectories until done
#         episodes_sum_of_rewards = []
#         log_probs = []
#         returns = []
#         for ep in range(self.nb_episodes):
#             x,_ = env.reset()
#             rewards = []
#             episode_cum_reward = 0
#             while(True):
#                 a, log_prob = self.sample_action_and_log_prob(x)
#                 y,r,d,_,_ = env.step(a)
#                 log_probs.append(log_prob)
#                 rewards.append(r)
#                 episode_cum_reward += r
#                 x=y
#                 if d:
#                     # compute returns-to-go
#                     new_returns = []
#                     G_t = 0
#                     for r in reversed(rewards):
#                         G_t = r + self.gamma * G_t
#                         new_returns.append(G_t)
#                     new_returns = list(reversed(new_returns))
#                     returns.extend(new_returns)
#                     episodes_sum_of_rewards.append(episode_cum_reward)
#                     break
#         # make loss
#         returns = torch.tensor(returns)
#         log_probs = torch.cat(log_probs)
#         loss = -(returns * log_probs).mean()
#         # gradient step
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return np.mean(episodes_sum_of_rewards)

#     def train(self, env, nb_rollouts):
#         avg_sum_rewards = []
#         for ep in trange(nb_rollouts):
#             avg_sum_rewards.append(self.one_gradient_step(env))
#         return avg_sum_rewards