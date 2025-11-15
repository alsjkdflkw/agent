"""
Test trained DQN agent against Q-Learning agent from main.py

Loads trained agents (READ-ONLY) and runs tournament comparison.
Does NOT overwrite any saved models.
"""

import os
import joblib
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.agents import QLearningAgent, RandomAgent, BaseAgent
from uno.tournament import Tournament


class GymnasiumToBaseAgentWrapper(BaseAgent):
    """
    Wrapper to make GymnasiumAgent compatible with Tournament class.
    
    Tournament class expects BaseAgent interface (select_action with state, legal_actions).
    GymnasiumAgent expects observation dict.
    """
    
    def __init__(self, gymnasium_agent):
        self.agent = gymnasium_agent
    
    def select_action(self, state, legal_actions):
        """
        Convert Tournament's interface to GymnasiumAgent's interface.
        
        Args:
            state: Raw state from Tournament
            legal_actions: List of legal actions
            
        Returns:
            Selected action index
        """
        # GymnasiumAgent expects observation dict
        return self.agent.select_action(state, legal_actions)


def load_dqn_agent(model_path, env):
    """Load DQN agent (read-only)."""
    if not os.path.exists(model_path):
        print(f"[!] DQN model not found: {model_path}")
        print("[*] Train first with: python fast_train.py --episodes 1000")
        print("[*] Or: python evolutionary_train.py --generations 100")
        return None
    
    agent = DQNAgent(env.action_space, env.observation_space)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration for testing
    
    print(f"[✓] Loaded DQN agent from {model_path}")
    print(f"    Episodes trained: {agent.episodes}")
    print(f"    Total steps: {agent.steps}")
    return agent


def load_qlearning_agent(model_path):
    """Load Q-Learning agent (read-only)."""
    if not os.path.exists(model_path):
        print(f"[!] Q-Learning model not found: {model_path}")
        print("[*] Run main.py first to train Q-Learning agent")
        return None
    
    agent = joblib.load(model_path)
    agent.eval_mode()  # Set to evaluation mode
    
    print(f"[✓] Loaded Q-Learning agent from {model_path}")
    print(f"    Q-table size: {len(agent.q_table)}")
    return agent


def test_against_qlearning(dqn_path='evolution/best_agent.pkl', 
                           qlearning_path='q_learning_agent.pkl',
                           num_games=100):
    """
    Test DQN agent against Q-Learning agent.
    
    READ-ONLY: Does not modify any saved models.
    """
    print("\n" + "="*70)
    print("TESTING: DQN Agent vs Q-Learning Agent")
    print("="*70 + "\n")
    
    env = UnoEnv(render_mode=None)
    
    # Load DQN agent
    dqn_agent = load_dqn_agent(dqn_path, env)
    if dqn_agent is None:
        return
    
    # Load Q-Learning agent
    qlearning_agent = load_qlearning_agent(qlearning_path)
    if qlearning_agent is None:
        return
    
    print(f"\n[*] Running {num_games} game tournament...")
    print("[*] This is READ-ONLY - no models will be modified\n")
    
    # Wrap DQN agent for compatibility with Tournament class
    wrapped_dqn = GymnasiumToBaseAgentWrapper(dqn_agent)
    
    # Run tournament
    tournament = Tournament(env, [wrapped_dqn, qlearning_agent])
    payoffs = tournament.run_tournament(num_games)
    
    print("\n" + "="*70)
    print("TOURNAMENT COMPLETE")
    print("="*70)


def test_against_random(dqn_path='evolution/best_agent.pkl', num_games=100):
    """
    Test DQN agent against Random agent.
    
    READ-ONLY: Does not modify any saved models.
    """
    print("\n" + "="*70)
    print("TESTING: DQN Agent vs Random Agent")
    print("="*70 + "\n")
    
    env = UnoEnv(render_mode=None)
    
    # Load DQN agent
    dqn_agent = load_dqn_agent(dqn_path, env)
    if dqn_agent is None:
        return
    
    # Create random opponent
    random_agent = RandomAgent()
    
    print(f"\n[*] Running {num_games} game tournament...")
    print("[*] This is READ-ONLY - no models will be modified\n")
    
    # Wrap DQN agent for compatibility with Tournament class
    wrapped_dqn = GymnasiumToBaseAgentWrapper(dqn_agent)
    
    # Run tournament
    tournament = Tournament(env, [wrapped_dqn, random_agent])
    payoffs = tournament.run_tournament(num_games)
    
    print("\n" + "="*70)
    print("TOURNAMENT COMPLETE")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test trained DQN agent (READ-ONLY, no overwriting)'
    )
    
    parser.add_argument('--dqn-model', type=str, 
                       default='evolution/best_agent.pkl',
                       help='Path to DQN model (default: evolution/best_agent.pkl)')
    parser.add_argument('--qlearning-model', type=str,
                       default='q_learning_agent.pkl',
                       help='Path to Q-Learning model (default: q_learning_agent.pkl)')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of test games (default: 100)')
    parser.add_argument('--vs-random', action='store_true',
                       help='Test against random agent instead of Q-Learning')
    parser.add_argument('--fast-model', action='store_true',
                       help='Use fast_train model (models/agent.pkl)')
    
    args = parser.parse_args()
    
    # Override model path if fast model requested
    if args.fast_model:
        args.dqn_model = 'models/agent.pkl'
    
    if args.vs_random:
        test_against_random(args.dqn_model, args.games)
    else:
        test_against_qlearning(args.dqn_model, args.qlearning_model, args.games)


if __name__ == '__main__':
    main()
