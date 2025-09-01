"""
start with an intial policy and iteratiuvely update the policy to find the best one
1) Policy Evaluation: Evaluate the given policy
2) Policy Improvment: refind or find a better policy
"""

"""
    Policy evaluation
    estiamtes the state-value function for a given policy in a MDP. 
    uses the bellman expectation equation for policy evaluation

pseudo-code:

    Function PolicyEvaluation(policy, transition_probs, rewards, gamma, eps):
    Initialize V for each state to 0
    Repeat:
        new_V = copy of V
        For each state s:
            action = policy's action for state s
            state_value = 0
            For each possible next state s' and corresponding reward r:
                Calculate expected value for s' using transition probabilities and rewards
                Update state_value using the calculated expected value
            Update new_V for state s with state_value
        If max change between new_V and V < eps:
            Break
        V = new_V
    Return V
"""

def PolicyEvaluation(policy, transition_matrix, reward_matrix, gamma, theta, states):

    # intialise V with arbitrary value
    V = {state: 0 for state in states}

    # until convergence
    while True:
        new_V = V.copy()

        #update each state's state-value function based on bellman expectation equation
        for state in states:
            action = policy[state]

            #intialise state's value function and delta to 0
            delta = 0
            state_value = 0

            # compute the state's expected value given the policy's action
            for next_state in states:
                transition_prob = transition_matrix[state][action][next_state] 
                reward = reward_matrix[state][action]

                #update the state's value function based off this succesor state
                state_value += transition_prob * (reward + gamma*V[next_state])
            
            #Assign state's value to new value function
            new_V[state] = state_value

            #update delta 
            delta = max(delta, abs(new_V[state] - V[state]))

        #check convergence
        if delta < theta:
            break
        
        #update Value function with the new state values for next iteration
        V = new_V
    
    return V

"""
Policy Improvement
    refine or enhance existing policy given a state-value function V

    Objective:
        make the policy greedy with respect to current V 
            - the action at each state that maximises the return based on the current value function is chosen

    Pseudo Code:
        Function PolicyImprovement(V, transition_probs, rewards, actions, gamma):
        Initialize an empty policy new_policy
        For each state s:
            Initialize best_action to None and best_value to negative infinity
            For each action a available in state s:
                Initialize action_value to 0
                For each possible next state s' and corresponding reward r:
                    Calculate expected value for s' based on transition probabilities, rewards, and V
                    Update action_value with the calculated expected value
                If action_value is greater than best_value:
                    Update best_value with action_value
                    Update best_action with action a
            Update new_policy for state s with best_action
        Return new_policy
"""

def PolicyImprovement(V, transition_matrix, reward_matrix, actions, gamma, states):
    new_policy={}

    # iterate through each estate to update its policy base on action
    for state in states:

        # Initialise values for best action and best value for state s
        best_action = None
        best_value = float("-inf")

        # compute through each action to find the one with greatest expected return
        for action in actions[state]:
            action_value = 0

            # calculate the expected value of this action over all successor states
            for next_state in states: 
                transition_prob = transition_matrix[state][action][next_state]
                reward = reward_matrix[state][action]
                action_value += transition_prob * (reward + (gamma * V[next_state]))

            # Update best action if this action's value is higher 
            if action_value > best_value:
                best_value = action_value
                best_action = action

        # update improved/refined action to policy for this state
        new_policy[state] = best_action

    return new_policy

"""
Policy Iteration
    combining the two to determine optimal policy and correlated state-value function
    Alternates between policy evaluation and policy improvement until no change in policies after improvement step
    indicating convergence to the optimal policy

    policy evaluation: compute the state-value function, for the current policy 
    policy Improvement: update the policy by making it "greedy" with respect to the current state value function

    pseudo code
        Function PolicyIteration(states, actions, transition_probs, rewards, gamma, eps):
        Initialize an arbitrary policy (e.g., the first available action for each state)
        Repeat:
            Evaluate the current policy to get the state-value function V
            Improve the policy based on V to get new_policy
            If new_policy is the same as the current policy:
                Break
            Update the current policy to new_policy
        Return the final state-value function V and the optimal policy
"""

def policyIteration(states, actions, transition_matrix, reward_matrix, gamma=0.9, theta=1e-3, track_history=False):
    policy = {state: actions[state][0] for state in states}
    iteration_count = 0
    value_history = [] if track_history else None

    while True:
        iteration_count += 1

        #compute the state value function for current policy
        V = PolicyEvaluation(policy, transition_matrix, reward_matrix, gamma, theta, states)
        
        if track_history:
            value_history.append(V.copy())
        
        print(f"Policy Iteration {iteration_count}: Value Function after Policy Evaluation: {V}")

        #improve policy given its current state value function
        new_policy = PolicyImprovement(V, transition_matrix, reward_matrix, actions, gamma, states)

        print(f"Policy Iteration {iteration_count}: Policy after Improvement: {new_policy} \n")

        # Check convergence / optimal policy
        if new_policy == policy:
            break

        #update improved policy
        policy = new_policy

    if track_history:
        return V, policy, value_history, iteration_count
    else:
        return V, policy


