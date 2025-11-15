"""
Simple example demonstrating how to create and test a Gymnasium API agent.

This script shows the most basic usage pattern for creating an agent
and testing it against opponents.
"""

from uno.env import UnoEnv
from uno.gymnasium_agent import GymnasiumAgent, RandomGymnasiumAgent
import numpy as np


class SimpleAgent(GymnasiumAgent):
    """
    A simple example agent that demonstrates the Gymnasium API.
    
    Strategy:
    - Avoid drawing cards when possible
    - Prefer action cards over number cards
    - Play the highest value card available
    """
    
    def select_action(self, observation, legal_actions):
        """
        Select action based on simple heuristics.
        
        Args:
            observation: Current game state
            legal_actions: List of legal action indices
            
        Returns:
            int: Selected action
        """
        # Remove draw action if we have other options
        playable_actions = [a for a in legal_actions if a != 60]
        
        if not playable_actions:
            # Must draw a card
            return 60
        
        # Prefer action cards (indices 40-59) over number cards (0-39)
        action_cards = [a for a in playable_actions if a >= 40]
        if action_cards:
            # Return highest value action card
            return max(action_cards)
        
        # Return highest value number card
        return max(playable_actions)


def main():
    """Main demonstration function."""
    print("\n" + "="*80)
    print("SIMPLE GYMNASIUM AGENT EXAMPLE")
    print("="*80 + "\n")
    
    # Create the Uno environment
    print("1. Creating Uno environment...")
    env = UnoEnv(render_mode=None)  # Use render_mode="human" to see gameplay
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}\n")
    
    # Create agents
    print("2. Creating agents...")
    my_agent = SimpleAgent(env.action_space, env.observation_space)
    opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    print("   SimpleAgent (Player 0)")
    print("   RandomAgent (Player 1)\n")
    
    # Play a single game
    print("3. Playing a single game...\n")
    obs, info = env.reset()
    done = False
    turn_count = 0
    
    agents = [my_agent, opponent]
    
    while not done:
        current_player = info["player_id"]
        legal_actions = info["legal_actions"]
        
        # Select and execute action
        agent = agents[current_player]
        action = agent.select_action(obs, legal_actions)
        
        obs, reward, done, truncated, info = env.step(action)
        turn_count += 1
    
    # Get results
    payoffs = env.env.get_payoffs()
    winner = np.argmax(payoffs)
    
    print(f"   Game completed in {turn_count} turns")
    print(f"   Winner: Player {winner}")
    print(f"   Payoffs: Player 0: {payoffs[0]}, Player 1: {payoffs[1]}\n")
    
    # Run a tournament
    print("4. Running a 100-game tournament...\n")
    wins = [0, 0]
    
    for game in range(100):
        obs, info = env.reset()
        done = False
        
        while not done:
            current_player = info["player_id"]
            legal_actions = info["legal_actions"]
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, truncated, info = env.step(action)
        
        payoffs = env.env.get_payoffs()
        winner = np.argmax(payoffs)
        wins[winner] += 1
        
        if (game + 1) % 25 == 0:
            print(f"   Progress: {game + 1}/100 games completed")
    
    # Print results
    print("\n" + "="*80)
    print("TOURNAMENT RESULTS")
    print("="*80)
    print(f"SimpleAgent wins: {wins[0]}/100 ({wins[0]}%)")
    print(f"RandomAgent wins: {wins[1]}/100 ({wins[1]}%)")
    print("="*80 + "\n")
    
    # Show how to save/load agent
    print("5. Saving agent to file...")
    import joblib
    joblib.dump(my_agent, 'simple_agent.pkl')
    print("   Agent saved to: simple_agent.pkl")
    
    print("\n6. Loading agent from file...")
    loaded_agent = joblib.load('simple_agent.pkl')
    print("   Agent loaded successfully!\n")
    
    print("="*80)
    print("EXAMPLE COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("- Modify SimpleAgent strategy to improve performance")
    print("- Try different heuristics and compare results")
    print("- Implement a learning agent with the learn() method")
    print("- Read TESTING_INSTRUCTIONS.md for more details")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
