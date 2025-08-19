import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_split(file_path):
    """
    Create two subplots: left for individual agent rewards, right for total reward
    """
    epochs = []
    reward_0 = []
    reward_1 = []
    reward_2 = []
    reward_total = []
    
    # Read data
    with open(file_path, 'r') as f:
        # Skip header line
        next(f)
        
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:
                epochs.append(int(data[0]))
                reward_0.append(float(data[1]))
                reward_1.append(float(data[2]))
                reward_2.append(float(data[3]))
                reward_total.append(float(data[4]))
    
    # Create two wide subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle('Multi-Agent SAC Training Progress', fontsize=22, fontweight='bold')
    
    # Left subplot - individual agent reward curves
    ax1.plot(epochs, reward_0, 'b-', linewidth=2, label='Agent 0')
    ax1.plot(epochs, reward_1, 'r-', linewidth=2, label='Agent 1')
    ax1.plot(epochs, reward_2, 'g-', linewidth=2, label='Agent 2')
    ax1.set_title('Individual Agent Rewards', fontsize=22)
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Reward Sum', fontsize=20)
    ax1.legend(fontsize=18, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(epochs))
    ax1.set_ylim(min(min(reward_0), min(reward_1), min(reward_2)) * 1.5, max(max(reward_0), max(reward_1), max(reward_2)) * 1.5)
    
    # Calculate average rewards for each agent
    avg_0 = np.mean(reward_0)
    avg_1 = np.mean(reward_1)
    avg_2 = np.mean(reward_2)
    
    # Add average lines to left subplot
    ax1.axhline(y=avg_0, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_1, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_2, color='green', linestyle='--', alpha=0.5)
    
    # Right subplot - total reward curve
    ax2.plot(epochs, reward_total, 'm-', linewidth=3)
    ax2.set_title('Total Reward (All Agents)', fontsize=22)
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Total Reward Sum', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))
    ax2.set_ylim(min(reward_total) * 1.5, max(reward_total) * 1.5)
    
    # Add total reward statistics
    avg_total = np.mean(reward_total)
    max_total = np.max(reward_total)
    ax2.axhline(y=avg_total, color='gray', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_total:.0f}')
    ax2.axhline(y=max_total, color='gray', linestyle='-', alpha=0.7, 
                label=f'Max: {max_total:.0f}')
    ax2.legend(fontsize=18)
    
    # Add scatter points for better visualization
    for ax, data in [(ax1, [reward_0, reward_1, reward_2]), (ax2, [reward_total])]:
        for i, y in enumerate(data):
            color = ['blue', 'red', 'green', 'magenta'][i]
            ax.scatter(epochs, y, color=color, s=40, alpha=0.6)
    
    # Add statistics text box
    stat_text = f"Training Statistics:\n" \
                f"Epochs: {len(epochs)}\n" \
                f"Avg Agent 0: {avg_0:.0f}\n" \
                f"Avg Agent 1: {avg_1:.0f}\n" \
                f"Avg Agent 2: {avg_2:.0f}\n" \
                f"Avg Total: {avg_total:.0f}\n" \
                f"Max Total: {max_total:.0f}"
    fig.text(0.98, 0.15, stat_text, fontsize=18, 
             bbox=dict(facecolor='white', alpha=0.8), ha='right')
    
    # Save and show
    plt.tight_layout()
    plt.savefig('./img/masac_rewards_split.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Training progress: {len(epochs)} epochs")
    print(f"Agent 0 - Average: {avg_0:.2f}")
    print(f"Agent 1 - Average: {avg_1:.2f}")
    print(f"Agent 2 - Average: {avg_2:.2f}")
    print(f"Total reward - Average: {avg_total:.2f}, Max: {max_total:.2f}")

if __name__ == "__main__":
    file_path = "./data/ma_epoch_rewards.txt"
    plot_rewards_split(file_path)