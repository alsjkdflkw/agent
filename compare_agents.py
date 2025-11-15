"""
Compare DQN Agent vs Q-Learning Agent (from main.py)

This script allows you to test the DQN agent against the built-in 
Q-Learning agent to see which performs better.
"""

import argparse
import os
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.agents import QLearningAgent, RandomAgent
from uno.gymnasium_agent import create_gymnasium_agent_wrapper
from test_gymnasium_agent import run_tournament
import joblib


def load_or_train_qlearning_agent(episodes=100, force_retrain=False):
    """
    Load or train a Q-Learning agent.
    
    Args:
        episodes: Number of training episodes if training
        force_retrain: Force retraining even if saved model exists
        
    Returns:
        Trained QLearningAgent wrapped in Gymnasium interface
    """
    model_path = "q_learning_agent.pkl"
    
    # Check if trained model exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing Q-Learning agent from {model_path}...")
        agent = joblib.load(model_path)
        agent.eval_mode()  # Set to evaluation mode
        print(f"Q-Learning agent loaded (epsilon: {agent.epsilon})")
    else:
        print(f"Training new Q-Learning agent for {episodes} episodes...")
        print("(This may take a while...)")
        
        env = UnoEnv(render_mode=None)
        agent = QLearningAgent()
        opponent = RandomAgent()
        
        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                current_player_id = info["player_id"]
                legal_actions = info["legal_actions"]
                
                if current_player_id == 0:
                    action = agent.select_action(obs, legal_actions)
                else:
                    action = opponent.select_action(obs, legal_actions)
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                if current_player_id == 0:
                    agent.learn(obs, action, reward, next_obs, done, episode)
                
                obs = next_obs
            
            if (episode + 1) % 20 == 0:
                print(f"  Episode {episode + 1}/{episodes} - Epsilon: {agent.epsilon:.3f}")
        
        # Save the trained agent
        joblib.dump(agent, model_path)
        print(f"Q-Learning agent saved to {model_path}")
        agent.eval_mode()
    
    # Wrap in Gymnasium interface
    WrappedQLearningAgent = create_gymnasium_agent_wrapper(QLearningAgent)
    
    # Create wrapper instance and copy the trained agent's data
    class PretrainedQLearningAgent:
        def __init__(self, trained_agent):
            self.agent = trained_agent
        
        def select_action(self, observation, legal_actions):
            return self.agent.select_action(observation, legal_actions)
    
    wrapped = PretrainedQLearningAgent(agent)
    wrapped.select_action = lambda obs, legal: agent.select_action(obs, legal)
    
    return wrapped


def compare_agents(dqn_model_path=None, qlearning_episodes=100, num_games=100, 
                   force_retrain_qlearning=False):
    """
    Compare DQN agent against Q-Learning agent.
    
    Args:
        dqn_model_path: Path to trained DQN model (None = untrained)
        qlearning_episodes: Episodes to train Q-Learning agent if needed
        num_games: Number of comparison games
        force_retrain_qlearning: Force retrain Q-Learning agent
    """
    print("\n" + "="*80)
    print("DQN AGENT vs Q-LEARNING AGENT COMPARISON")
    print("="*80 + "\n")
    
    # Setup environment
    env = UnoEnv(render_mode=None)
    
    # Load/create DQN agent
    print("1. Setting up DQN Agent...")
    dqn_agent = DQNAgent(env.action_space, env.observation_space)
    
    if dqn_model_path and os.path.exists(dqn_model_path):
        print(f"   Loading DQN agent from {dqn_model_path}")
        dqn_agent.load(dqn_model_path)
        print(f"   DQN agent loaded (episodes: {dqn_agent.episodes}, steps: {dqn_agent.steps})")
    else:
        print("   Using untrained DQN agent")
    
    dqn_agent.epsilon = 0.0  # No exploration during comparison
    
    # Load/train Q-Learning agent
    print("\n2. Setting up Q-Learning Agent...")
    qlearning_agent = load_or_train_qlearning_agent(
        episodes=qlearning_episodes,
        force_retrain=force_retrain_qlearning
    )
    
    # Run comparison
    print(f"\n3. Running comparison tournament ({num_games} games)...")
    print("="*80)
    
    agents = [dqn_agent, qlearning_agent]
    agent_names = ["DQN Agent", "Q-Learning Agent"]
    
    # Run tournament
    stats = run_tournament(env, agents, num_games=num_games, verbose=False)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for i, name in enumerate(agent_names):
        print(f"{name}:")
        print(f"  Wins: {int(stats['wins'][i])}/{num_games} ({stats['win_rates'][i]:.1f}%)")
        print(f"  Avg Payoff: {stats['avg_payoffs'][i]:.3f}")
    
    print(f"\nAverage game length: {stats['avg_turns']:.1f} turns")
    
    # Determine winner
    print("\n" + "="*80)
    if stats['win_rates'][0] > stats['win_rates'][1]:
        margin = stats['win_rates'][0] - stats['win_rates'][1]
        print(f"🏆 DQN Agent WINS by {margin:.1f}%!")
    elif stats['win_rates'][1] > stats['win_rates'][0]:
        margin = stats['win_rates'][1] - stats['win_rates'][0]
        print(f"🏆 Q-Learning Agent WINS by {margin:.1f}%!")
    else:
        print("🤝 TIE! Both agents performed equally well.")
    print("="*80 + "\n")
    
    return stats


def quick_comparison():
    """Quick comparison with untrained agents."""
    print("\n" + "="*80)
    print("QUICK COMPARISON - Untrained Agents")
    print("="*80)
    print("Comparing DQN and Q-Learning agents without prior training")
    print("Both agents will use their default initialization\n")
    
    compare_agents(
        dqn_model_path=None,
        qlearning_episodes=0,  # Use default Q-Learning agent
        num_games=50,
        force_retrain_qlearning=False
    )


def trained_comparison(dqn_path, qlearning_episodes=100, num_games=100):
    """Compare trained agents."""
    print("\n" + "="*80)
    print("TRAINED AGENTS COMPARISON")
    print("="*80)
    print(f"DQN Model: {dqn_path}")
    print(f"Q-Learning: {qlearning_episodes} episodes training\n")
    
    compare_agents(
        dqn_model_path=dqn_path,
        qlearning_episodes=qlearning_episodes,
        num_games=num_games,
        force_retrain_qlearning=False
    )


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Compare DQN agent vs Q-Learning agent from main.py'
    )
    
    parser.add_argument('--dqn-model', type=str, default=None,
                       help='Path to trained DQN model (default: untrained)')
    parser.add_argument('--qlearning-episodes', type=int, default=100,
                       help='Episodes to train Q-Learning agent (default: 100)')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of comparison games (default: 100)')
    parser.add_argument('--retrain-qlearning', action='store_true',
                       help='Force retrain Q-Learning agent even if saved')
    parser.add_argument('--quick', action='store_true',
                       help='Quick comparison with untrained agents')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_comparison()
    else:
        compare_agents(
            dqn_model_path=args.dqn_model,
            qlearning_episodes=args.qlearning_episodes,
            num_games=args.games,
            force_retrain_qlearning=args.retrain_qlearning
        )


if __name__ == '__main__':
    main()
