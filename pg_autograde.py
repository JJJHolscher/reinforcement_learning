import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        action_probs = torch.Tensor([[pred[i, a]] for i, a in enumerate(actions)])
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    obs = env.reset()
    done = False
    while not done:
        obs = torch.Tensor(obs)

        states.append(obs)
        action = policy.sample_action(obs)
        obs, reward, done, _ = env.step(int(action))

        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states_2t = torch.Tensor(len(states), len(states[0]))
    for i in range(len(states_2t)):
        states_2t[i] = states[i]

    actions_2t = torch.Tensor(len(actions), len(actions[0]))
    torch.cat(actions, out=actions_2t)
    return states_2t, torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(dones)

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    print(states)
    print(actions)
    probs = policy.get_probs(states, actions)
    print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        print(pred.shape, actions.shape)
        action_probs = torch.Tensor([[pred[i, a]] for i, a in enumerate(actions)])
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    obs = env.reset()
    done = False
    while not done:
        obs = torch.Tensor(obs)

        states.append(obs)
        action = policy.sample_action(obs)
        obs, reward, done, _ = env.step(int(action))

        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states_2t = torch.Tensor(len(states), len(states[0]))
    for i in range(len(states_2t)):
        states_2t[i] = states[i]

    actions_2t = torch.Tensor(len(actions), len(actions[0]))
    torch.cat(actions, out=actions_2t)
    return states_2t, torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(dones)

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    print(states)
    print(actions)
    probs = policy.get_probs(states, actions)
    print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        actions_1 = torch.zeros_like(pred)
        actions_1[:, 0] = (actions - 1) * -1
        actions_1[:, 1] = actions
        action_probs = torch.sum(pred * actions_1, dim=1)
        print(pred.shape, actions.shape)
        print(action_probs)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        actions = torch.squeeze(actions)

        actions_1 = torch.zeros_like(pred)
        actions_1[:, 0] = (actions - 1) * -1
        actions_1[:, 1] = actions
        action_probs = torch.sum(pred * actions_1, dim=1)
        print(pred.shape, actions.shape)
        print(action_probs)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        actions = torch.squeeze(actions)

        actions_1 = torch.zeros_like(pred)
        actions_1[:, 0] = (actions - 1) * -1
        actions_1[:, 1] = actions
        action_probs = torch.sum(pred * actions_1, dim=1)
        print(action_probs)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        actions = torch.squeeze(actions)

        actions_1 = torch.zeros_like(pred)
        actions_1[:, 0] = (actions - 1) * -1
        actions_1[:, 1] = actions
        action_probs = torch.sum(pred * actions_1, dim=1, keepdim=True)
        print(action_probs)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    obs = env.reset()
    done = False
    while not done:
        obs = torch.Tensor(obs)

        states.append(obs)
        action = policy.sample_action(obs)
        obs, reward, done, _ = env.step(int(action))

        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states_2t = torch.Tensor(len(states), len(states[0]))
    for i in range(len(states_2t)):
        states_2t[i] = states[i]

    actions_2t = torch.Tensor(len(actions), len(actions[0]))
    torch.cat(actions, out=actions_2t)
    return states_2t, torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(dones)

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    print(states)
    print(actions)
    probs = policy.get_probs(states, actions)
    print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return self.softmax(self.l2(self.relu(self.l1(x))))
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        pred = self.forward(obs)
        actions = torch.squeeze(actions)

        actions_1 = torch.zeros_like(pred)
        actions_1[:, 0] = (actions - 1) * -1
        actions_1[:, 1] = actions
        
        action_probs = torch.sum(pred * actions_1, dim=1, keepdim=True)
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self.forward(obs), 1)

        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    obs = env.reset()
    done = False
    while not done:
        obs = torch.Tensor(obs)

        states.append(obs)
        action = policy.sample_action(obs)
        obs, reward, done, _ = env.step(int(action))

        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states_2t = torch.Tensor(len(states), len(states[0]))
    for i in range(len(states_2t)):
        states_2t[i] = states[i]

    actions_2t = torch.Tensor(len(actions), len(actions[0]))
    torch.cat(actions, out=actions_2t)
    return states_2t, torch.LongTensor(actions), torch.Tensor(rewards), torch.Tensor(dones)

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    print(states)
    print(actions)
    probs = policy.get_probs(states, actions)
    print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    # print(states)
    # print(actions)
    probs = policy.get_probs(states, actions)
    # print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    # print(states)
    # print(actions)
    probs = policy.get_probs(states, actions)
    print(probs)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    probs = policy.get_probs(states, actions)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = G @ torch.log(probs)
    print(loss)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    states, actions, rewards, _ = episode
    probs = policy.get_probs(states, actions)
    
    g = 0
    G = [discount_factor * g + rewards[t] for t in range(len(states) - 1, -1, -1)]
    G.reverse()
    G = torch.Tensor(G).unsqueeze(0)

    loss = - G @ torch.log(probs)
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    for i in range(num_episodes):

        optimizer.zero_grad()
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations
