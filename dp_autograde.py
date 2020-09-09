import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    
    
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    # loop door alle states heen 
        # sla de oude state value op 
        # Bereken de nieuwe state value door de SOM (kans omhoog * loop over waar je terrecht kunt komen * reward) kans omlaag..
    # kijk of je nog door moet gaan of stoppen
    delta = 1000 
    while delta > theta:
    # for x in range(2):
        delta = 0
        
#   loop throw possible states
        for state in range(env.nS):
            old_state_value = V[state]
            new_state_value = 0

    #       loop shrow possible actions in state
            for action in range(env.nA):

                # print("kans omhoog", policy[state][action])
                # print("kans omhoog uitkomen", env.P[state][action][0][0])
                # print("direct reward",env.P[state][action][0][2] )
                # print("value of that new state", discount_factor * V[env.P[state][action][0][1]] )

                current_state_value = policy[state][action] * env.P[state][action][0][0] * ( env.P[state][action][0][2] + ( discount_factor * V[env.P[state][action][0][1]] ) ) 
#                 print("current state value", current_state_value)
                new_state_value += current_state_value
                
            delta = max(delta, abs(old_state_value - new_state_value))
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    policy_stable = False
    while not policy_stable:
        V = policy_eval_v(policy, env, discount_factor=discount_factor)

        policy_stable = True
        for state in range(env.nS):
            old_best_action = np.argmax(policy[state])
            best_action = -1
            best_value = -float('inf')
            for action in range(env.nA):
                value = V[env.P[state][action][0][1]]
                if value > best_value:
                    best_value = value
                    best_action = action
            
            for action in range(env.nA):
                if action == best_action:
                    policy[state][action] = 1
                else:
                    policy[state][action] = 0
            
            if best_action != old_best_action:
                policy_stable = False

    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    
    delta = float('inf')
    while delta > theta:
        delta = 0

        for state in range(env.nS):
            for action in range(env.nA):
                prob, new_state, reward, _ = env.P[state][action][0]
                old_state_value = Q[state][action]

                Q[state][action] = prob * (reward + discount_factor * max(Q[new_state]))
                delta = max(delta, abs(old_state_value - Q[state][action]))

    policy = np.expand_dims(np.max(Q, axis=1), axis=1) @ np.ones((1, Q.shape[1]))
    policy = (policy == Q).astype(float)
    policy /= np.expand_dims(np.sum(policy, axis=1), axis=1)
    return policy, Q
