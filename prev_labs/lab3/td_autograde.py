import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

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
        if np.random.random() >= self.epsilon:
            action = int(np.argmax(self.Q[obs]))
        
        else:
            action = int(np.random.randint(0, len(self.Q[obs])))
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        s_0 = env.reset()
        a_0 = policy.sample_action(s_0)
        s_1, r, done, info = env.step(a_0)

        i = 1
        R = r

        while not done:
            a_1 = policy.sample_action(s_1)
            Q[s_0, a_0] += alpha * (r + discount_factor * Q[s_1, a_1]
                                      - Q[s_0, a_0])
            a_0, s_0 = a_1, s_1
            s_1, r, done, info = env.step(a_0)

            i += 1
            R += r
        
        Q[s_0, a_0] += alpha * (r - Q[s_0, a_0])        
        stats.append((i, R))

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        s_0 = env.reset()
        done = False

        while not done:
            a = policy.sample_action(s_0)
            s_1, r, done, info = env.step(a)

            max_Q = np.amax(Q[s_1]) if not done else 0
            Q[s_0, a] += alpha * (r + discount_factor * max_Q
                                    - Q[s_0, a])
            s_0 = s_1
            i += 1
            R += r
        
        stats.append((i, R))
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
