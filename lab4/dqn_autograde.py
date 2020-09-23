import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
#         print("JAAJAJA", x.shape)
        relu = nn.ReLU()
    
        output_firt_lin_layer = self.l1(x)
#         print("OUT", output_firt_lin_layer.shape)
        output_relu = F.relu(output_firt_lin_layer)
        end =  self.l2(output_relu)
#         print(end.shape)

        return end

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    
    if it < 1000:
        slope = -0.00095

        epsilon = 1 + (it * slope)

        
    else:
        epsilon = 0.05
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """

#       just a starting state ([-0.0119,  0.0057, -0.0113,  0.0341])
        obs = torch.tensor(obs).float()

        with torch.no_grad():
            prediction = self.Q.forward(obs)
#         print("PREDICITON", prediction)
        prediction = prediction.numpy()

        coin = np.random.uniform()
#         print(coin)
#         print(1- self.epsilon)
#         print("len", len(prediction))
        
        # take greedy action
        if coin <= 1 - self.epsilon:
            action = np.argmax(prediction)
            
        
            
        #else select a random action
        else:
#             print("JAAJJA")
            action = np.random.randint(low = 0, high = len(prediction) )


        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

    
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # YOUR CODE HERE
            raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
        loss = train(Q, memory, optimizer, batch_size, discount_factor)
        print("loss", los)
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", los)
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            True = False
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    print("transitions", transitions)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    print("transitions", transitions)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    print("transitions", transitions)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    print(transitions.shape)
    print("transitions", transitions)

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    print(type(transitions))
    print(transitions.shape)
    print("transitions", transitions)

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    print("IK BEN ERIN")
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    print("IK BEN ERIN")
    print(len(memory))
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    print(len(transitions))
    print("transitions", transitions)
    print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    print(memory.capacity)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    print("mem", memory)
    print(memory.capacity)
    print(len(memory))
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    print("IK BEN ERIN")
    print(len(memory))
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
#         print("MEEMMM", self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
#     print("mem", memory)
#     print(memory.capacity)
    print(len(memory))
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
#     print("mem", memory)
#     print(memory.capacity)
    print("LEN MEM", len(memory))
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
            break
            
#             # YOUR CODE HERE
#             raise NotImplementedError
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    print("IK BEN ERIN")
    print(len(memory))
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)
    print("loss", loss)
    print(loss.shape)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)

    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
#             break

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)
    

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)

    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss", loss)
        
#             break

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)

    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            print("loss")
        
#             break

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
    print(discounted_values_incl_dones)
    print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)
    

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    network_output = Q(states)

    computed_q_vals = torch.gather(network_output, 1, actions)

    return computed_q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """


    dones = dones.to(torch.long).squeeze()

#   zoek op welke index nummers in done de waarde 1 heeft > dus welke in terminal state komen
    index_nrs = (dones == 1).nonzero().squeeze()

    network_output = Q(next_states)

#   pak de goeie netwerk output waarden
    computed_target_values = torch.gather(network_output, 1, rewards.to(torch.long))
    
#   doe de multiply er tegenaan
    discounted_target_values = discount_factor * computed_target_values
    discounted_target_values = discounted_target_values.squeeze()
    

#   zet de waarde 0 op de index waarden waarbij je in de terminal state komt
    discounted_values_incl_dones = discounted_target_values.scatter(0, index_nrs, 0).unsqueeze(1)
    
#     print(discounted_values_incl_dones)
#     print(discounted_values_incl_dones.shape)
    
    return discounted_values_incl_dones

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

#     print(len(transitions))
#     print("transitions", transitions)
#     print("HAALOOO")

    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)
    

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)

    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        print(i)
        
        state = env.reset()
        
        for x in range(batch_size):
    
            # fill memory with transitions
            s = env.reset()

            # 0 or 1 .. push left or right
            a = env.action_space.sample()

            s_next, r, done, _ = env.step(a)

            # Push a transition
            memory.push((s, a, r, s_next, done))

        
        steps = 0
        while True:
            
#           sample transitions uit memory
#           krijg de predictions uit je netwerk
#           bereken de loss
#           first fill the memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
#             print("loss")
        

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations
