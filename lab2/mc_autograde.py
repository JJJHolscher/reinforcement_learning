import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    
#     print("object", object)
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    
    def get_probs(self, state, actions):


#         print("state", state)
#         print("actions", actions)
        """
        je krijgt dus de state binnen en dan alle mogelijk acties voor die state. 
        Hit or stick. Je moet dan per actie de kans bepalen dat je die actie gaat uitvoeren
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        
        probs = []

# loop door de states
# bekijke wat de state waarde 
# loop door acties
# als je actie hit zou zijn obv state en dat is ook de actie die je binnen krijgt > zet prob op 100
# anders zet prob op 0


        for action in actions:
#             print("state[0]", state[0])
            hand_sum = state[0][0]
#             print("HAND SUM", hand_sum)
            
#             print("current state", hand_sum)
#             print("current action", action)
                
#               nu wil je stoppen dus action = 0
            if hand_sum < 20:
                recommended_action = 1

#                   als wat je zou aanbevelen en wat de gebruiker wilt overeenkomen, prob > 100 procent
                if recommended_action == action:

                    probs.append(1)
                    
                else:
                    probs.append(0)


            if hand_sum == 20 or hand_sum == 21:
                recommended_action = 0

                if recommended_action == action:

                    probs.append(1)
                else:
                    probs.append(0)

                        
#         print("END PROBS", probs)
        return np.array(probs)
    
    def sample_action(self, state):
#         print("current state", state)
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
#       hit
        if state[0] < 20:
            action = 1

#       stick
        if state[0] == 20 or state[0] == 21:
            action = 0

    
        
        return action
     

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    
    states = []
    actions = []
    rewards = []
    dones = []
    
    done = False


    # begin met een starting state en bepaal actie
    new_state = env.reset()
    states.append(new_state)
#     print("start", new_state)
    
    #voer acties uit en zolang je niet door bent blijf je doorgaan 
    while done != True:
        
        action = policy.sample_action(new_state)
#         print("ACTION", action)
        actions.append(action)

#         print("ACTION HIER", action)
        try:
            if len(action) == 1:
                action = action[0]
            
        except:
            None
        new_env = env.step(action)
#         print("new", new_env)
        
        done = new_env[2]
        dones.append(done)

        
        
        new_state = new_env[0]
        
        # voeg alleen states toe als je niet in de terminate state bent gekomen .. dit zegt de hint
        if done == False:
            states.append(new_state)
        
        
        reward = new_env[1]
        rewards.append(reward)
        

    return states, actions, rewards, dones


def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = (float)
    
    # initialize all state values arbirarly
    # initialize an empty list for all states
    
    
    for i in tqdm(range(num_episodes)):
        data = sample_episode(env, policy)
#         print(data)
        visited_states = data[0]
        
        # voeg nu elke state die je bent tegengekomen toe aan je dict
        for state in visited_states:
            
            # als state nog niet in V zit voeg toe
            if state not in V:
                V[state] = []

        g = 0
#         loop reversed throw the trajectory
#         print("state", data[0])
        data[0].reverse()
        data[2].reverse()

        #loop door de trajectory
        for state, reward in ( zip(data[0], data[2]) ):
            
            #bereken de waarde van die state door de directe reward van het spel te pakken. Deze waarde geldt dan voor hele trajectory
            g += discount_factor * reward
#             print("G", g)
            
            # update nu de state uit de trajectory met deze reward
            current_list = V[state]
#             print("CURRENT LIST", current_list)
            current_list.append(g)
#             print("UPDATED", current_list)
            V[state] = current_list


#     print("@@@", V)
    for key, value in V.items():
        V[key] = sum(value) / len(value)
#         print(key, value)
        
    #loop door alle states in de dict
    # pak de lijst per state
    # avg de lijst en vervang daarmee de huidige lijst voor die stata

    
    return V

# num_episodes=10000
V_10k = mc_prediction(env, SimpleBlackjackPolicy(), num_episodes=10000)
print(V_10k)

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
#         print("STATES", states)
#         print("ACTIONS", actions)
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        probs = len(states) * [0.5]
#         print("probs", probs)
        
        
        
        
        return probs
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        #       hit
        action = np.random.randint(low=0.0, high=2.0, size = 1)
#         print("action", action)
        

            
        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    V = defaultdict(float)
    returns_count = (float)
    
    # initialize all state values arbirarly
    # initialize an empty list for all states

#   loop door episodes
    for i in tqdm(range(num_episodes)):
        data = sample_episode(env, policy)
#         print("data", data)

        visited_states = data[0]
        
        # voeg nu elke state die je bent tegengekomen toe aan je dict
        for state in visited_states:
            
            # als state nog niet in V zit voeg toe
            if state not in V:
                V[state] = []

        g = 0
        target_policy_prob = 1
        behavioural_policy_prob = 1
        
        
#       loop reversed throw the trajectory, rewards and actions
        data[0].reverse() # states
        data[2].reverse() # rewards
        data[1].reverse() # actions
        
#         print("states", data[0])
#         print("actions", data[1])
#         print("rewards", data[2])

        
        #loop door de trajectory in reversed order
        for state, reward, action in ( zip(data[0], data[2], data[1]) ):
#             print("state", state)
#             print("reward", reward)
#             print("action", action)
            
            target_policy_prob *= target_policy.get_probs([state], [0, 1])[action[0]] # [0 , 1] prob for stick and prob for hit
            behavioural_policy_prob *= behavior_policy.get_probs(state, [0, 1])[action[0]]
#             behavioural_policy_prob *= behavior_policy.get_probs(state)[0]
#             print("tar", target_policy_prob)
#             print("beh", behavioural_policy_prob)
           
            weights = target_policy_prob/behavioural_policy_prob
#             print("weights", weights)
#             print("REWARD", reward)
            
            #bereken de waarde van die state door de directe reward van het spel te pakken. Deze waarde geldt dan voor hele trajectory
            g += discount_factor * reward * weights
#             print("G", g)
            
            # update nu de state uit de trajectory met deze reward
            current_list = V[state]
#             print("CURRENT LIST", current_list)
            current_list.append(g)
#             print("UPDATED", current_list)
            V[state] = current_list


#     print("@@@", V)
    for key, value in V.items():
        V[key] = sum(value) / len(value)
#         print(key, value)
        
    #loop door alle states in de dict
    # pak de lijst per state
    # avg de lijst en vervang daarmee de huidige lijst voor 
    

    # Keeps track of current V and count of returns for each state
    # to calculate an update.

    
    # YOUR CODE HERE
    
    return V

V_10k = mc_importance_sampling(env, RandomBlackjackPolicy(), SimpleBlackjackPolicy(), num_episodes=10)
print(V_10k)
