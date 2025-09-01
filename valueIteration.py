"""
Value Iteration


"""

"""
State-Value Computation using Bellman Optimality Equation

pseudo code:
    Function ComputeStateValue(s, v, transition_probs, rewards, gamma):
    For each action a available in state s:
        For each possible next state s' and reward r:
            Calculate expected return using:
                Sum over transition_prob * (reward + gamma * value_of_next_state)
    Return the highest expected return among all actions
"""

def computeStateValue(state, V, transition_matrix, reward_matrix, actions, gamma, states):
    #stores expected values for each action in state s
    expected_values = []

    # iterate through available actions in state s
    for action in actions[state]:
        action_value = 0

        # compute the expected value for the action by summing over all successor states
        for next_state in states:
            transition_prob = transition_matrix[state][action][next_state]
            reward = reward_matrix[state][action]

            # update the action's expected value ussing bellman equation
            action_value += transition_prob * (reward + (gamma*V[next_state]))
        
        expected_values.append(action_value)

    #return the hisghest expected value among all actions
    return max(expected_values)

"""
Policy Extraction based on value function

    determining the best action for each state based on a given state value function V
    uses bellman's optimality equation

    objective:
        make the policy "greedy" with respect to current value function

    Pseudo code:
        Function ExtractPolicy(V, transition_probs, rewards, actions, gamma):
        Initialize an empty policy
        For each state s:
            Initialize best_action to None and best_action_value to negative infinity
            For each action a available in state s:
                Initialize action_value to 0
                For each possible next state s':
                    Update action_value using transition probabilities, rewards, and V
                If action_value is greater than best_action_value:
                    Update best_action_value with action_value
                    Update best_action with action a
            Update the policy for state s with best_action
        Return the policy
"""

def extractPolicy(V, transition_matrix, reward_matrix, actions, gamma, states):
    #initialise empty policy
    policy={}

    # iterate through each state in state space
    for state in states:

        best_action = None
        best_action_value = float("-inf")

        # evaluate each possible action to find the best one for current state
        for action in actions[state]:
            action_value = 0

            # compute expected value for the state
            for next_state in states:
                transition_prob = transition_matrix[state][action][next_state]
                reward = reward_matrix[state][action]

                # update the action value using components of the Bellman equation
                action_value += transition_prob * (reward + (gamma*V[next_state]))
            
            # update the best action if this action's value is higher
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        
        #append the computed best action to the policy
        policy[state] = best_action

    return policy


"""
Value Iteration Algorithm

    compute the optimal value function and, subsequently the optimal policy for a MDP.
    It iteratively updates the value function until it converges and then extracts the policy based on that "converged" value function

    Pseudo Code:
        Function ValueIteration(states, actions, transition_probs, rewards, gamma, eps):
        Initialize V for each state to 0
        Repeat:
            new_V = V.copy()
            For each state:
                Update new_V[state] using Bellman Optimality Equation
            If max change between new_V and V < eps:
                Break
            V = new_V
        Extract policy based on V using the policy extraction method
        Return V, policy
"""

def valueIteration(states, actions, transition_matrix, reward_matrix, gamma=0.9, theta=1e-3, track_history=False):
    V = {state: 0 for state in states}
    iteration = 0
    value_history = [V.copy()] if track_history else None

    while True:
        new_V = V.copy()
        delta = 0

        # Compute State-value function
        for state in states:
            new_V[state] = computeStateValue(state, V, transition_matrix, reward_matrix, actions, gamma, states)
            delta = max(delta, abs(new_V[state] - V[state]))
        
        iteration += 1
        
        if track_history:
            value_history.append(new_V.copy())
            
        print(f"Value Iteration {iteration}: Value Function: {new_V}")
        print(f"Value Iteration {iteration}: Delta: {delta}\n")
        
        #Check Convergence 
        if delta < theta:
            break

        V = new_V
    
    # extract policy from state-value function
    policy = extractPolicy(V, transition_matrix, reward_matrix, actions, gamma, states)

    if track_history:
        return V, policy, value_history, iteration
    else:
        return V, policy
