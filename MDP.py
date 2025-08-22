# General MDP 
class MDP:
    def __init__(self, states, actions, transition_matrix, reward_matrix, discount_factor=1.0):
        """
        Initialize the MDP with given states, actions, transition probabilities, rewards, and discount factor.

        Parameters:
        - states: List of states in the MDP
        - actions: List of actions available in the MDP
        - transition_matrix: Matrix where each row represents the current state, each column represents an action,
                             and the inner lists represent the next state probabilities.
        - reward_matrix: Matrix where each row represents the current state and each column represents an action.
        - discount_factor: Discount factor for future rewards (gamma in Sutton & Barto)
        """
        self.states = states
        self.actions = actions
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor

    # Converting the Transition Prob, rewards and actions to Dictionary.
    def convert_to_dictionary(self):
        """
        Convert transition matrix and reward matrix to a dictionary format which is more intuitive for certain operations.

        Returns:
        - transition_probs: Dictionary of transition probabilities
        - rewards: Dictionary of rewards for state-action pairs
        - actions: Dictionary of available actions for each state
        """
        # Convert actions list to dictionary format
        actions = {state: [act for act in self.actions] for state in self.states}

        # Initialize the transition_probs and rewards dictionaries
        transition_probs = {s: {} for s in self.states}
        rewards = {s: {} for s in self.states}

        for i, s in enumerate(self.states):
            for j, a in enumerate(self.actions):
                transition_probs[s][a] = {}
                for k, s_prime in enumerate(self.states):
                    # Set the transition probability for s' from the matrix
                    transition_probs[s][a][s_prime] = self.transition_matrix[i][k]

                    # Set the reward for action a in state s from the matrix
                    rewards[s][a] = self.reward_matrix[i][j]

        return transition_probs, rewards, actions