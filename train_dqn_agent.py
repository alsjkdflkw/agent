"""
Training script for DQN Agent

This script trains a Deep Q-Network agent on the Uno game environment.
It includes options for long-term training with progress tracking and
automatic model checkpointing.
"""

import argparse
import os
import time
from datetime import datetime
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.gymnasium_agent import RandomGymnasiumAgent
from test_gymnasium_agent import run_tournament
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_training_progress(stats_history, save_path='training_progress.png'):
    """
    Plot training metrics over time.
    
    Args:
        stats_history: List of stats dictionaries
        save_path: Path to save the plot
    """
    episodes = [s['episodes'] for s in stats_history]
    epsilons = [s['epsilon'] for s in stats_history]
    losses = [s['avg_loss'] for s in stats_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot epsilon decay
    ax1.plot(episodes, epsilons, 'b-', label='Epsilon')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon')
    ax1.set_title('Exploration Rate Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot loss
    ax2.plot(episodes, losses, 'r-', label='Avg Loss')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def train_dqn_agent(num_episodes=1000,
                    eval_interval=100,
                    eval_games=20,
                    save_interval=500,
                    model_dir='models',
                    verbose=True):
    """
    Train a DQN agent on the Uno environment.
    
    Args:
        num_episodes: Number of training episodes
        eval_interval: Episodes between evaluations
        eval_games: Number of games per evaluation
        save_interval: Episodes between model saves
        model_dir: Directory to save models
        verbose: Whether to print detailed progress
    """
    print("="*80)
    print("DQN AGENT TRAINING")
    print("="*80)
    print(f"Training for {num_episodes} episodes")
    print(f"Evaluation every {eval_interval} episodes")
    print(f"Model saves every {save_interval} episodes")
    print()
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize environment and agents
    env = UnoEnv(render_mode=None)
    
    # Create DQN agent with optimized hyperparameters
    dqn_agent = DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=32,
        target_update_freq=100,
        hidden_size=128
    )
    
    # Opponent for training
    opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    agents = [dqn_agent, opponent]
    
    # Training statistics
    stats_history = []
    win_rates = []
    episode_rewards = []
    start_time = time.time()
    
    print("Starting training...")
    print("-"*80)
    
    # Training loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        turn_count = 0
        
        while not done:
            current_player = info['player_id']
            legal_actions = info['legal_actions']
            
            # Select agent
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Learn from experience (only DQN agent)
            if current_player == 0:
                dqn_agent.learn(obs, action, reward, next_obs, done, info)
                episode_reward += reward
            
            obs = next_obs
            turn_count += 1
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            stats = dqn_agent.get_stats()
            avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Epsilon: {stats['epsilon']:.3f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {stats['avg_loss']:.4f}")
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION at Episode {episode + 1}")
            print(f"{'='*80}")
            
            # Temporarily disable exploration for evaluation
            original_epsilon = dqn_agent.epsilon
            dqn_agent.epsilon = 0.0
            
            # Run tournament
            stats = run_tournament(env, agents, num_games=eval_games, verbose=False)
            win_rate = stats['win_rates'][0]
            win_rates.append(win_rate)
            
            print(f"DQN Agent Win Rate: {win_rate:.1f}%")
            print(f"Average Game Length: {stats['avg_turns']:.1f} turns")
            
            # Restore epsilon
            dqn_agent.epsilon = original_epsilon
            
            # Save stats
            agent_stats = dqn_agent.get_stats()
            agent_stats['win_rate'] = win_rate
            stats_history.append(agent_stats)
            
            print(f"{'='*80}\n")
        
        # Save model checkpoint
        if (episode + 1) % save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f'dqn_agent_ep{episode+1}_{timestamp}.pkl')
            dqn_agent.save(model_path)
            print(f"Model checkpoint saved: {model_path}")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes: {num_episodes}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Episodes per minute: {num_episodes/(elapsed_time/60):.1f}")
    
    # Final evaluation
    print("\nFinal Evaluation (100 games):")
    dqn_agent.epsilon = 0.0  # Disable exploration
    final_stats = run_tournament(env, agents, num_games=100, verbose=False)
    print(f"Final Win Rate: {final_stats['win_rates'][0]:.1f}%")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'dqn_agent_final.pkl')
    dqn_agent.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Plot training progress
    if stats_history:
        plot_path = os.path.join(model_dir, 'training_progress.png')
        plot_training_progress(stats_history, plot_path)
    
    # Save training summary
    summary_path = os.path.join(model_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"DQN Agent Training Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Episodes: {num_episodes}\n")
        f.write(f"  Time: {elapsed_time/60:.1f} minutes\n")
        f.write(f"  Learning Rate: {dqn_agent.learning_rate}\n")
        f.write(f"  Discount Factor: {dqn_agent.discount_factor}\n")
        f.write(f"  Batch Size: {dqn_agent.batch_size}\n")
        f.write(f"  Buffer Capacity: {dqn_agent.replay_buffer.buffer.maxlen}\n\n")
        f.write(f"Final Performance:\n")
        f.write(f"  Win Rate: {final_stats['win_rates'][0]:.1f}%\n")
        f.write(f"  Episodes Trained: {dqn_agent.episodes}\n")
        f.write(f"  Total Steps: {dqn_agent.steps}\n")
    
    print(f"Training summary saved: {summary_path}")
    print("="*80 + "\n")
    
    return dqn_agent, stats_history


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Train DQN Agent for Uno')
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=100,
                       help='Episodes between evaluations (default: 100)')
    parser.add_argument('--eval-games', type=int, default=20,
                       help='Games per evaluation (default: 20)')
    parser.add_argument('--save-interval', type=int, default=500,
                       help='Episodes between saves (default: 500)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for saving models (default: models)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Run training
    train_dqn_agent(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        save_interval=args.save_interval,
        model_dir=args.model_dir,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
