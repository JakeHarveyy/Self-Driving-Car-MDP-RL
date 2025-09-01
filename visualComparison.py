"""
Algorithm Comparison with Visual State-Value Tracking
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from policyIteration import policyIteration
from valueIteration import valueIteration

def plot_value_evolution(pi_history, vi_history, states, save_path=None):
    """
    Plot the evolution of state values during iterations for both algorithms.
    
    Parameters:
    - pi_history: Value function history from Policy Iteration
    - vi_history: Value function history from Value Iteration
    - states: List of states
    - save_path: Optional path to save the plot
    """
    # Filter out terminal states for cleaner visualization
    non_terminal_states = [s for s in states if s not in ['Destination Reached', 'Acident']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Policy Iteration Plot
    ax1.set_title('Policy Iteration: State Value Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('State Value')
    ax1.grid(True, alpha=0.3)
    
    for state in non_terminal_states:
        values = [v_func[state] for v_func in pi_history]
        ax1.plot(range(len(values)), values, marker='o', linewidth=2, label=state)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Value Iteration Plot
    ax2.set_title('Value Iteration: State Value Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('State Value')
    ax2.grid(True, alpha=0.3)
    
    for state in non_terminal_states:
        values = [v_func[state] for v_func in vi_history]
        ax2.plot(range(len(values)), values, marker='o', linewidth=2, label=state)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.close()  # Close instead of show for non-interactive

def plot_convergence_comparison(pi_history, vi_history, states, save_path=None):
    """
    Plot convergence speed comparison between algorithms.
    """
    # Calculate maximum change per iteration
    pi_deltas = []
    for i in range(1, len(pi_history)):
        max_delta = max(abs(pi_history[i][state] - pi_history[i-1][state]) for state in states)
        pi_deltas.append(max_delta)
    
    vi_deltas = []
    for i in range(1, len(vi_history)):
        max_delta = max(abs(vi_history[i][state] - vi_history[i-1][state]) for state in states)
        vi_deltas.append(max_delta)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(pi_deltas) + 1), pi_deltas, 'b-o', linewidth=2, label='Policy Iteration', markersize=6)
    plt.semilogy(range(1, len(vi_deltas) + 1), vi_deltas, 'r-s', linewidth=2, label='Value Iteration', markersize=4)
    
    plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Value Change (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Convergence plot saved to {save_path}")
    
    plt.close()  # Close instead of show

def visual_algorithm_comparison(states, actions, transition_matrix, reward_matrix, 
                              start_state='Clear Road', gamma=0.9, theta=1e-3, save_plots=False):
    """
    Complete visual comparison of algorithms with multiple plots.
    
    Parameters:
    - save_plots: If True, saves plots to files
    """
   
    # Run algorithms with history tracking
    start_time = time.time()
    pi_values, pi_policy, pi_history, pi_iterations = policyIteration(
        states, actions, transition_matrix, reward_matrix, gamma, theta, track_history=True
    )
    pi_time = time.time() - start_time
    
    print("🎯 Value Iteration with tracking...")
    start_time = time.time()
    vi_values, vi_policy, vi_history, vi_iterations = valueIteration(
        states, actions, transition_matrix, reward_matrix, gamma, theta, track_history=True
    )
    vi_time = time.time() - start_time
    
    # Display basic comparison
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Policy Iteration: {pi_iterations} iterations, {pi_time:.4f}s")
    print(f"Value Iteration:  {vi_iterations} iterations, {vi_time:.4f}s")
    print(f"Speed Ratio:      {vi_iterations/pi_iterations:.1f}x more iterations for VI")
    
    # Check policy agreement
    policy_matches = sum(1 for state in states if pi_policy[state] == vi_policy[state])
    print(f"Policy Agreement: {policy_matches}/{len(states)} states match")
    
    # Generate plots
    print(f"\n📈 Generating visualizations...")
    
    # 1. Value evolution plot
    plot_value_evolution(pi_history, vi_history, states, 
                        "value_evolution.png" if save_plots else None)
    
    # 2. Convergence comparison
    plot_convergence_comparison(pi_history, vi_history, states,
                               "convergence_comparison.png" if save_plots else None)
    
    
    return {
        "policy_iteration": {
            "values": pi_values,
            "policy": pi_policy,
            "history": pi_history,
            "iterations": pi_iterations,
            "time": pi_time
        },
        "value_iteration": {
            "values": vi_values,
            "policy": vi_policy,
            "history": vi_history,
            "iterations": vi_iterations,
            "time": vi_time
        }
    }

def animate_value_evolution(history, states, algorithm_name, save_path=None):
    """
    Create an animated plot showing value evolution (optional advanced feature).
    """
    from matplotlib.animation import FuncAnimation
    
    non_terminal_states = [s for s in states if s not in ['Destination Reached', 'Acident']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        values = [history[frame][state] for state in non_terminal_states]
        bars = ax.bar(range(len(non_terminal_states)), values, color='skyblue', alpha=0.7)
        
        ax.set_title(f'{algorithm_name}: Iteration {frame}', fontsize=14, fontweight='bold')
        ax.set_xlabel('States')
        ax.set_ylabel('State Values')
        ax.set_xticks(range(len(non_terminal_states)))
        ax.set_xticklabels(non_terminal_states, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set consistent y-axis limits
        max_val = max(max(v_func[state] for state in non_terminal_states) for v_func in history)
        min_val = min(min(v_func[state] for state in non_terminal_states) for v_func in history)
        ax.set_ylim(min_val - 10, max_val + 10)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    anim = FuncAnimation(fig, animate, frames=len(history), interval=500, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
        print(f"🎬 Animation saved to {save_path}")
    
    plt.show()
    return anim
