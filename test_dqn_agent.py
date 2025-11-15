"""
Test and demonstrate the DQN learning agent.

This script allows you to:
1. Train a DQN agent quickly
2. Test a trained agent
3. Compare performance before and after training
"""

from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.gymnasium_agent import RandomGymnasiumAgent, HeuristicGymnasiumAgent
from test_gymnasium_agent import run_tournament
import os


def quick_train_demo(episodes=100):
    """
    Quick training demonstration (100 episodes).
    
    Shows how the agent improves over time.
    """
    print("\n" + "="*80)
    print("QUICK TRAINING DEMO - DQN Agent")
    print("="*80)
    print(f"Training for {episodes} episodes...\n")
    
    # Setup
    env = UnoEnv(render_mode=None)
    dqn_agent = DQNAgent(env.action_space, env.observation_space)
    opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    # Evaluate before training
    print("1. Testing BEFORE training (20 games):")
    dqn_agent.epsilon = 0.0  # No exploration for evaluation
    stats_before = run_tournament(env, [dqn_agent, opponent], num_games=20, verbose=False)
    print(f"   Win Rate: {stats_before['win_rates'][0]:.1f}%\n")
    
    # Train
    print(f"2. Training for {episodes} episodes...")
    dqn_agent.epsilon = 1.0  # Reset exploration
    agents = [dqn_agent, opponent]
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            current_player = info['player_id']
            legal_actions = info['legal_actions']
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            if current_player == 0:
                dqn_agent.learn(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
        
        if (episode + 1) % 20 == 0:
            stats = dqn_agent.get_stats()
            print(f"   Episode {episode + 1}/{episodes} - "
                  f"Epsilon: {stats['epsilon']:.3f}, "
                  f"Loss: {stats['avg_loss']:.4f}")
    
    # Evaluate after training
    print("\n3. Testing AFTER training (20 games):")
    dqn_agent.epsilon = 0.0  # No exploration for evaluation
    stats_after = run_tournament(env, [dqn_agent, opponent], num_games=20, verbose=False)
    print(f"   Win Rate: {stats_after['win_rates'][0]:.1f}%\n")
    
    # Summary
    improvement = stats_after['win_rates'][0] - stats_before['win_rates'][0]
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Before Training: {stats_before['win_rates'][0]:.1f}% wins")
    print(f"After Training:  {stats_after['win_rates'][0]:.1f}% wins")
    print(f"Improvement:     {improvement:+.1f}%")
    print(f"Episodes:        {dqn_agent.episodes}")
    print(f"Steps:           {dqn_agent.steps}")
    print("="*80 + "\n")
    
    # Save model
    model_path = 'dqn_agent_demo.pkl'
    dqn_agent.save(model_path)
    print(f"Model saved to: {model_path}")
    print("Load it later with: agent.load('dqn_agent_demo.pkl')\n")
    
    return dqn_agent


def test_trained_agent(model_path, num_games=100):
    """
    Test a trained DQN agent.
    
    Args:
        model_path: Path to saved model
        num_games: Number of test games
    """
    print("\n" + "="*80)
    print(f"TESTING TRAINED AGENT: {model_path}")
    print("="*80 + "\n")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Load agent
    env = UnoEnv(render_mode=None)
    dqn_agent = DQNAgent(env.action_space, env.observation_space)
    dqn_agent.load(model_path)
    dqn_agent.epsilon = 0.0  # No exploration during testing
    
    print(f"Model loaded: {model_path}")
    print(f"Episodes trained: {dqn_agent.episodes}")
    print(f"Total steps: {dqn_agent.steps}\n")
    
    # Test against Random agent
    print(f"1. Testing vs Random Agent ({num_games} games):")
    random_opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    stats_random = run_tournament(env, [dqn_agent, random_opponent], 
                                  num_games=num_games, verbose=False)
    print(f"   DQN Win Rate: {stats_random['win_rates'][0]:.1f}%")
    print(f"   Random Win Rate: {stats_random['win_rates'][1]:.1f}%\n")
    
    # Test against Heuristic agent
    print(f"2. Testing vs Heuristic Agent ({num_games} games):")
    heuristic_opponent = HeuristicGymnasiumAgent(env.action_space, env.observation_space)
    stats_heuristic = run_tournament(env, [dqn_agent, heuristic_opponent],
                                    num_games=num_games, verbose=False)
    print(f"   DQN Win Rate: {stats_heuristic['win_rates'][0]:.1f}%")
    print(f"   Heuristic Win Rate: {stats_heuristic['win_rates'][1]:.1f}%\n")
    
    print("="*80 + "\n")


def compare_agents():
    """Compare DQN agent with other agents."""
    print("\n" + "="*80)
    print("AGENT COMPARISON")
    print("="*80 + "\n")
    
    env = UnoEnv(render_mode=None)
    
    # Create agents
    random_agent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    heuristic_agent = HeuristicGymnasiumAgent(env.action_space, env.observation_space)
    dqn_agent = DQNAgent(env.action_space, env.observation_space)
    dqn_agent.epsilon = 0.0
    
    num_games = 50
    
    # Random vs Heuristic
    print(f"1. Random vs Heuristic ({num_games} games):")
    stats1 = run_tournament(env, [random_agent, heuristic_agent], 
                           num_games=num_games, verbose=False)
    print(f"   Random: {stats1['win_rates'][0]:.1f}%, Heuristic: {stats1['win_rates'][1]:.1f}%\n")
    
    # DQN vs Random
    print(f"2. DQN (untrained) vs Random ({num_games} games):")
    stats2 = run_tournament(env, [dqn_agent, random_agent],
                           num_games=num_games, verbose=False)
    print(f"   DQN: {stats2['win_rates'][0]:.1f}%, Random: {stats2['win_rates'][1]:.1f}%\n")
    
    # DQN vs Heuristic
    print(f"3. DQN (untrained) vs Heuristic ({num_games} games):")
    stats3 = run_tournament(env, [dqn_agent, heuristic_agent],
                           num_games=num_games, verbose=False)
    print(f"   DQN: {stats3['win_rates'][0]:.1f}%, Heuristic: {stats3['win_rates'][1]:.1f}%\n")
    
    print("="*80)
    print("Note: Train the DQN agent to see improvement!")
    print("="*80 + "\n")


def main():
    """Main menu for DQN testing."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'quick-train':
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            quick_train_demo(episodes)
        
        elif command == 'test':
            if len(sys.argv) < 3:
                print("Usage: python test_dqn_agent.py test <model_path> [num_games]")
                return
            model_path = sys.argv[2]
            num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 100
            test_trained_agent(model_path, num_games)
        
        elif command == 'compare':
            compare_agents()
        
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("DQN AGENT TESTING UTILITY")
    print("="*80)
    print("\nUsage:")
    print("  python test_dqn_agent.py quick-train [episodes]")
    print("      - Quick training demo (default: 100 episodes)")
    print()
    print("  python test_dqn_agent.py test <model_path> [num_games]")
    print("      - Test a trained model (default: 100 games)")
    print()
    print("  python test_dqn_agent.py compare")
    print("      - Compare different agents")
    print()
    print("Examples:")
    print("  python test_dqn_agent.py quick-train 200")
    print("  python test_dqn_agent.py test dqn_agent_demo.pkl 50")
    print("  python test_dqn_agent.py compare")
    print()
    print("For long-term training, use:")
    print("  python train_dqn_agent.py --episodes 5000")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
