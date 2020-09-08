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

        deltas = []
        
#   loop throw possible states
        for state in range(env.nS):
        
            old_state_value = V[state]

            new_state_value = 0

    #       loop shrow possible actions in state
            for action in range(env.nA):

                print("kans omhoog", policy[state][action])
                print("kans omhoog uitkomen", env.P[state][action][0][0])
                print("direct reward",env.P[state][action][0][2] )
                print("value of that new state", discount_factor * V[env.P[state][action][0][1]] )

                current_state_value = policy[state][action] * env.P[state][action][0][0] * ( env.P[state][action][0][2] + ( discount_factor * V[env.P[state][action][0][1]] ) ) 
#                 print("current state value", current_state_value)
                delta = abs(old_state_value - current_state_value)
                deltas.append(delta)
                
                V[state] += current_state_value
#                 print(V[state])
        delta = max(deltas)
#         print("delta", delta)

    return np.array(V)

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

                print("kans omhoog", policy[state][action])
                print("kans omhoog uitkomen", env.P[state][action][0][0])
                print("direct reward",env.P[state][action][0][2] )
                print("value of that new state", discount_factor * V[env.P[state][action][0][1]] )

                current_state_value = policy[state][action] * env.P[state][action][0][0] * ( env.P[state][action][0][2] + ( discount_factor * V[env.P[state][action][0][1]] ) ) 
#                 print("current state value", current_state_value)
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)

    return np.array(V)

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
    # while delta > theta:
    for x in range(10):
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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)

    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)

    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)

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
                delta = max(delta, abs(old_state_value - current_state_value))
                new_state_value += current_state_value
                
            V[state] = new_state_value
#                 print(V[state])
#         print("delta", delta)
    print("dome")
    return np.array(V)
