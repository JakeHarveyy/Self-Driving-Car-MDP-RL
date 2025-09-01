"""
Simple Algorithm Comparison Module for Policy Iteration vs Value Iteration
"""

import time
import numpy as np
from policyIteration import policyIteration
from valueIteration import valueIteration

def simulate_for_average_reward(states, transition_probs, rewards, start_state, policy, max_steps=100, num_episodes=1000):
    """
    Simulate the environment to compute the average reward over a number of episodes.

    Parameters:
    - states: List of states
    - transition_probs: Dictionary of transition probabilities
    - rewards: Dictionary of rewards for state-action pairs
    - start_state (Any): The initial state from which the simulation begins.
    - policy (dict): A mapping from states to actions representing the agent's policy.
    - max_steps (int, optional): Maximum number of steps for each episode. Defaults to 100.
    - num_episodes (int, optional): Number of episodes to simulate. Defaults to 1000.

    Returns:
    - float: The average reward accumulated per episode over the specified number of episodes.
    """
    total_rewards = 0
    terminal_states = {'Destination Reached', 'Acident'}
    
    for _ in range(num_episodes):
        current_state = start_state
        episode_reward = 0
        
        for _ in range(max_steps):
            if current_state in terminal_states:
                break
                
            action = policy[current_state]
            
            # Get reward for current state-action pair
            reward = rewards[current_state][action]
            episode_reward += reward
            
            # Sample the next state based on the transition probabilities
            prob_list = []
            for s_prime in states:
                prob_list.append(transition_probs[current_state][action][s_prime])
            next_state = np.random.choice(states, p=prob_list)
            
            current_state = next_state
            
        total_rewards += episode_reward
        
    return total_rewards / num_episodes

def compare_algorithms(states, actions, transition_probs, rewards, start_state='Clear Road', gamma=0.9, theta=1e-3):
    """
    Compare Policy Iteration and Value Iteration algorithms.
    
    Returns a dictionary with comparison results including timing and performance metrics.
    """
    print("🚗 Running Algorithm Comparison...")
    print("=" * 50)
    
    # Policy Iteration
    print("🔄 Policy Iteration...")
    start_time = time.time()
    PI_values, PI_policy = policyIteration(states, actions, transition_probs, rewards, gamma, theta)
    PI_time = time.time() - start_time
    PI_avg_reward = simulate_for_average_reward(states, transition_probs, rewards, start_state, PI_policy)
    
    print("🎯 Value Iteration...")
    # Value Iteration
    start_time = time.time()
    VI_values, VI_policy = valueIteration(states, actions, transition_probs, rewards, gamma, theta)
    VI_time = time.time() - start_time
    VI_avg_reward = simulate_for_average_reward(states, transition_probs, rewards, start_state, VI_policy)

    # Metrics
    value_diff = sum([abs(VI_values[state] - PI_values[state]) for state in states])
    policy_diff = sum([1 if VI_policy[state] != PI_policy[state] else 0 for state in states])

    return {
        "Value Iteration": {
            "Convergence Time": VI_time,
            "Value Function": VI_values,
            "Policy": VI_policy,
            "Average Reward": VI_avg_reward
        },
        "Policy Iteration": {
            "Convergence Time": PI_time,
            "Value Function": PI_values,
            "Policy": PI_policy,
            "Average Reward": PI_avg_reward
        },
        "Value Function Difference (Sum of Absolute Differences)": value_diff,
        "Policy Difference (Number of Different Actions)": policy_diff
    }

def display_comparison_results(results):
    """
    Display the comparison results in a nice formatted way.
    """
    pi_results = results["Policy Iteration"]
    vi_results = results["Value Iteration"]
    
    print("\n" + "=" * 60)
    print("📊 ALGORITHM COMPARISON RESULTS")
    print("=" * 60)
    
    # Timing Comparison
    print("\n⏱️  PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"Policy Iteration Time:    {pi_results['Convergence Time']:.4f} seconds")
    print(f"Value Iteration Time:     {vi_results['Convergence Time']:.4f} seconds")
    print(f"Speed Ratio (VI/PI):      {vi_results['Convergence Time']/pi_results['Convergence Time']:.2f}x")
    
    # Reward Comparison
    print(f"\nPolicy Iteration Avg Reward:  {pi_results['Average Reward']:.2f}")
    print(f"Value Iteration Avg Reward:   {vi_results['Average Reward']:.2f}")
    print(f"Reward Difference:            {abs(pi_results['Average Reward'] - vi_results['Average Reward']):.2f}")
    
    # Solution Quality
    print(f"\n🎯 SOLUTION QUALITY:")
    print("-" * 40)
    print(f"Value Function Difference:    {results['Value Function Difference (Sum of Absolute Differences)']:.6f}")
    print(f"Policy Differences:           {results['Policy Difference (Number of Different Actions)']} actions differ")
    print(f"Policy Match:                 {len(pi_results['Policy']) - results['Policy Difference (Number of Different Actions)']} / {len(pi_results['Policy'])} states")
    
    # Policy Comparison
    print(f"\n🚗 POLICY COMPARISON:")
    print("-" * 40)
    print(f"{'State':<20} {'Policy Iter':<15} {'Value Iter':<15} {'Match'}")
    print("-" * 40)
    
    for state in pi_results['Policy']:
        pi_action = pi_results['Policy'][state]
        vi_action = vi_results['Policy'][state]
        match = "✅" if pi_action == vi_action else "❌"
        print(f"{state:<20} {pi_action:<15} {vi_action:<15} {match}")
    
    # Value Function Comparison (Top 5)
    print(f"\n💰 VALUE FUNCTIONS (Top 5):")
    print("-" * 60)
    print(f"{'State':<20} {'Policy Iter':<15} {'Value Iter':<15} {'Difference'}")
    print("-" * 60)
    
    # Sort by Policy Iteration values
    sorted_states = sorted(pi_results['Value Function'].items(), key=lambda x: x[1], reverse=True)[:5]
    
    for state, pi_value in sorted_states:
        vi_value = vi_results['Value Function'][state]
        diff = abs(pi_value - vi_value)
        print(f"{state:<20} {pi_value:<15.3f} {vi_value:<15.3f} {diff:<15.6f}")
    
    print("\n" + "=" * 60)

def plot_convergence_comparison(results):
    """
    Simple text-based visualization of algorithm performance.
    """
    pi_time = results["Policy Iteration"]["Convergence Time"]
    vi_time = results["Value Iteration"]["Convergence Time"]
    
    print(f"\n📈 CONVERGENCE VISUALIZATION:")
    print("-" * 40)
    
    # Normalize times for visualization
    max_time = max(pi_time, vi_time)
    pi_bars = int((pi_time / max_time) * 30)
    vi_bars = int((vi_time / max_time) * 30)
    
    print(f"Policy Iteration:  {'█' * pi_bars}{' ' * (30 - pi_bars)} {pi_time:.4f}s")
    print(f"Value Iteration:   {'█' * vi_bars}{' ' * (30 - vi_bars)} {vi_time:.4f}s")
    
    # Policy agreement visualization
    total_states = len(results["Policy Iteration"]["Policy"])
    agreements = total_states - results["Policy Difference (Number of Different Actions)"]
    agreement_bars = int((agreements / total_states) * 30)
    
    print(f"\nPolicy Agreement:  {'█' * agreement_bars}{' ' * (30 - agreement_bars)} {agreements}/{total_states}")
