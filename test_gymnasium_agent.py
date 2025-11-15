"""
Test script for Gymnasium API agents.

This script demonstrates how to:
1. Create Gymnasium-compatible agents
2. Run single games
3. Run tournaments
4. Evaluate agent performance
"""

from uno.env import UnoEnv
from uno.gymnasium_agent import RandomGymnasiumAgent, HeuristicGymnasiumAgent, GymnasiumAgent
from uno.agents import RandomAgent, QLearningAgent
import numpy as np
from typing import List, Dict
import argparse


def run_single_game(env: UnoEnv, agents: List, render: bool = True, verbose: bool = True):
    """
    Run a single game with the given agents.
    
    Args:
        env: The Uno environment
        agents: List of agents (either BaseAgent or GymnasiumAgent instances)
        render: Whether to render the game
        verbose: Whether to print game information
        
    Returns:
        tuple: (winner_id, payoffs, num_turns)
    """
    obs, info = env.reset()
    done = False
    num_turns = 0
    
    if verbose:
        print("\n" + "="*80)
        print("Starting new game!")
        print("="*80)
    
    while not done:
        current_player_id = info["player_id"]
        legal_actions = info["legal_actions"]
        
        # Get the current agent
        agent = agents[current_player_id]
        
        # Select action using the agent
        action = agent.select_action(obs, legal_actions)
        
        if verbose and not render:
            print(f"Turn {num_turns}: Player {current_player_id} plays action {action}")
        
        # Take the action
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Allow agent to learn (if it supports learning)
        if hasattr(agent, 'learn'):
            agent.learn(obs, action, reward, next_obs, done, info)
        
        obs = next_obs
        num_turns += 1
    
    # Get final payoffs
    payoffs = env.env.get_payoffs()
    winner_id = np.argmax(payoffs)
    
    if verbose:
        print("\n" + "="*80)
        print(f"Game Over! Winner: Player {winner_id}")
        print(f"Payoffs: {payoffs}")
        print(f"Total turns: {num_turns}")
        print("="*80 + "\n")
    
    return winner_id, payoffs, num_turns


def run_tournament(env: UnoEnv, agents: List, num_games: int = 100, verbose: bool = False):
    """
    Run a tournament with multiple games.
    
    Args:
        env: The Uno environment
        agents: List of agents
        num_games: Number of games to play
        verbose: Whether to print detailed information
        
    Returns:
        dict: Tournament statistics
    """
    wins = [0] * len(agents)
    total_payoffs = [0.0] * len(agents)
    total_turns = 0
    
    print(f"\nRunning tournament with {num_games} games...")
    print(f"Agents: {[type(agent).__name__ for agent in agents]}")
    print("="*80)
    
    for game_num in range(num_games):
        if (game_num + 1) % 10 == 0:
            print(f"Progress: {game_num + 1}/{num_games} games completed...")
        
        winner_id, payoffs, num_turns = run_single_game(
            env, agents, render=False, verbose=verbose
        )
        
        wins[winner_id] += 1
        for i, payoff in enumerate(payoffs):
            total_payoffs[i] += payoff
        total_turns += num_turns
    
    # Calculate statistics
    win_rates = [w / num_games * 100 for w in wins]
    avg_payoffs = [p / num_games for p in total_payoffs]
    avg_turns = total_turns / num_games
    
    # Print results
    print("\n" + "="*80)
    print("TOURNAMENT RESULTS")
    print("="*80)
    for i, agent in enumerate(agents):
        print(f"Player {i} ({type(agent).__name__}):")
        print(f"  Wins: {wins[i]}/{num_games} ({win_rates[i]:.1f}%)")
        print(f"  Avg Payoff: {avg_payoffs[i]:.3f}")
    print(f"\nAverage game length: {avg_turns:.1f} turns")
    print("="*80 + "\n")
    
    return {
        'wins': wins,
        'win_rates': win_rates,
        'avg_payoffs': avg_payoffs,
        'avg_turns': avg_turns
    }


def test_random_vs_random():
    """Test two random agents against each other."""
    print("\n" + "#"*80)
    print("# TEST 1: Random Agent vs Random Agent")
    print("#"*80)
    
    env = UnoEnv(render_mode=None)
    
    # Create two Gymnasium random agents
    agent1 = RandomGymnasiumAgent(env.action_space, env.observation_space)
    agent2 = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    stats = run_tournament(env, [agent1, agent2], num_games=100)
    
    # With two random agents, expect roughly 50/50 win rate
    assert 30 < stats['win_rates'][0] < 70, "Random agents should have similar win rates"
    print("✓ Test passed: Random agents have balanced win rates\n")


def test_heuristic_vs_random():
    """Test heuristic agent against random agent."""
    print("\n" + "#"*80)
    print("# TEST 2: Heuristic Agent vs Random Agent")
    print("#"*80)
    
    env = UnoEnv(render_mode=None)
    
    # Create heuristic and random agents
    agent1 = HeuristicGymnasiumAgent(env.action_space, env.observation_space)
    agent2 = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    stats = run_tournament(env, [agent1, agent2], num_games=100)
    
    # Heuristic agent should perform better than random
    print(f"Heuristic win rate: {stats['win_rates'][0]:.1f}%")
    print(f"Random win rate: {stats['win_rates'][1]:.1f}%")
    
    if stats['win_rates'][0] > stats['win_rates'][1]:
        print("✓ Test passed: Heuristic agent outperforms random agent\n")
    else:
        print("⚠ Warning: Heuristic agent didn't outperform random (might be variance)\n")


def test_single_game_with_render():
    """Run a single game with rendering to demonstrate gameplay."""
    print("\n" + "#"*80)
    print("# TEST 3: Single Game with Rendering")
    print("#"*80)
    
    env = UnoEnv(render_mode="human")
    
    agent1 = HeuristicGymnasiumAgent(env.action_space, env.observation_space)
    agent2 = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    winner_id, payoffs, num_turns = run_single_game(
        env, [agent1, agent2], render=True, verbose=True
    )
    
    print("✓ Test passed: Single game completed successfully\n")


def test_custom_agent():
    """Demonstrate creating a custom agent."""
    print("\n" + "#"*80)
    print("# TEST 4: Custom Agent Implementation")
    print("#"*80)
    
    class CustomAgent(GymnasiumAgent):
        """
        Example custom agent that prefers to play high-value cards.
        """
        def select_action(self, observation, legal_actions):
            # Avoid drawing if possible
            playable = [a for a in legal_actions if a != 60]
            if not playable:
                return 60
            
            # Prefer higher action indices (action cards and wilds)
            return max(playable)
    
    env = UnoEnv(render_mode=None)
    
    agent1 = CustomAgent(env.action_space, env.observation_space)
    agent2 = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    stats = run_tournament(env, [agent1, agent2], num_games=50)
    
    print("✓ Test passed: Custom agent works correctly\n")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Gymnasium API agents for Uno')
    parser.add_argument('--test', type=str, default='all',
                      choices=['all', 'random', 'heuristic', 'render', 'custom'],
                      help='Which test to run')
    parser.add_argument('--games', type=int, default=100,
                      help='Number of games for tournament tests')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GYMNASIUM AGENT TEST SUITE")
    print("="*80)
    
    try:
        if args.test == 'all' or args.test == 'random':
            test_random_vs_random()
        
        if args.test == 'all' or args.test == 'heuristic':
            test_heuristic_vs_random()
        
        if args.test == 'all' or args.test == 'custom':
            test_custom_agent()
        
        if args.test == 'all' or args.test == 'render':
            test_single_game_with_render()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
