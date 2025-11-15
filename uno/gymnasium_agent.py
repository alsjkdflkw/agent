"""
Gymnasium API Agent Wrapper for Uno Game

This module provides a clean Gymnasium-compatible interface for creating
agents that can interact with the Uno environment.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import gymnasium as gym


class GymnasiumAgent(ABC):
    """
    Base class for Gymnasium-compatible agents.
    
    This class provides a standard interface that follows Gymnasium conventions
    for agent-environment interaction. Agents should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """
        Initialize the agent with action and observation spaces.
        
        Args:
            action_space: The action space of the environment (gym.spaces.Discrete)
            observation_space: The observation space of the environment
        """
        self.action_space = action_space
        self.observation_space = observation_space
        
    @abstractmethod
    def select_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Select an action given the current observation and legal actions.
        
        Args:
            observation: Dictionary containing the game state with keys:
                - 'obs': numpy array of shape (4, 4, 15) representing the game state
                - 'raw_obs': human-readable game information
                - 'legal_actions': dictionary of legal actions
                - 'raw_legal_actions': list of legal action names
                - 'action_record': history of actions
            legal_actions: List of legal action indices
            
        Returns:
            int: The selected action index (must be in legal_actions)
        """
        pass
    
    def reset(self):
        """
        Reset the agent's internal state (optional).
        
        This method is called at the beginning of each episode.
        Override if your agent maintains internal state that needs resetting.
        """
        pass
    
    def learn(self, 
              observation: Dict[str, Any], 
              action: int, 
              reward: float, 
              next_observation: Dict[str, Any], 
              done: bool,
              info: Dict[str, Any]) -> None:
        """
        Learn from experience (optional).
        
        This method is called after each step for learning agents.
        Override if your agent can learn from experience.
        
        Args:
            observation: The observation before taking the action
            action: The action that was taken
            reward: The reward received
            next_observation: The observation after taking the action
            done: Whether the episode is done
            info: Additional information from the environment
        """
        pass


class RandomGymnasiumAgent(GymnasiumAgent):
    """
    A random agent that selects actions uniformly from legal actions.
    
    This serves as a baseline and example implementation.
    """
    
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        super().__init__(action_space, observation_space)
        self.rng = np.random.RandomState()
    
    def select_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Randomly select from legal actions.
        
        Args:
            observation: Current game state (unused for random agent)
            legal_actions: List of legal action indices
            
        Returns:
            int: Randomly selected legal action
        """
        return self.rng.choice(legal_actions)
    
    def reset(self):
        """Reset random seed for reproducibility if needed."""
        pass


class HeuristicGymnasiumAgent(GymnasiumAgent):
    """
    A heuristic agent that uses simple rules to select actions.
    
    This agent demonstrates how to use the observation dictionary
    to make informed decisions.
    """
    
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        super().__init__(action_space, observation_space)
    
    def select_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Select action using simple heuristics.
        
        Strategy:
        1. Avoid drawing if possible
        2. Prioritize action cards (Skip, Reverse, Draw 2)
        3. Prioritize Wild cards if opponent has few cards
        4. Otherwise play the first legal card
        
        Args:
            observation: Current game state
            legal_actions: List of legal action indices
            
        Returns:
            int: Selected action based on heuristics
        """
        raw_obs = observation.get('raw_obs', {})
        
        # Remove draw action (60) if we have other options
        playable_actions = [a for a in legal_actions if a != 60]
        if not playable_actions:
            # Must draw
            return 60
        
        # Check opponent's card count (if available)
        num_cards = raw_obs.get('num_cards', [])
        opponent_has_few_cards = len(num_cards) > 1 and num_cards[1] <= 2
        
        # Priority 1: Play Wild Draw 4 if opponent is close to winning
        if opponent_has_few_cards:
            for action in playable_actions:
                if 56 <= action <= 59:  # Wild Draw 4
                    return action
        
        # Priority 2: Play Draw 2 cards
        for action in playable_actions:
            if 48 <= action <= 51:  # Draw 2
                return action
        
        # Priority 3: Play Skip cards
        for action in playable_actions:
            if 40 <= action <= 43:  # Skip
                return action
        
        # Priority 4: Play Reverse cards
        for action in playable_actions:
            if 44 <= action <= 47:  # Reverse
                return action
        
        # Priority 5: Play Wild cards
        for action in playable_actions:
            if 52 <= action <= 55:  # Wild
                return action
        
        # Default: Play the first legal card
        return playable_actions[0]


def create_gymnasium_agent_wrapper(base_agent_class):
    """
    Factory function to convert existing BaseAgent to GymnasiumAgent.
    
    This allows you to use existing agents with the new Gymnasium interface.
    
    Args:
        base_agent_class: A class that implements select_action(state, legal_actions)
        
    Returns:
        A wrapper class that conforms to GymnasiumAgent interface
    """
    class WrappedAgent(GymnasiumAgent):
        def __init__(self, action_space: gym.Space, observation_space: gym.Space, *args, **kwargs):
            super().__init__(action_space, observation_space)
            self.agent = base_agent_class(*args, **kwargs)
        
        def select_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
            return self.agent.select_action(observation, legal_actions)
        
        def learn(self, observation, action, reward, next_observation, done, info):
            if hasattr(self.agent, 'learn'):
                self.agent.learn(observation, action, reward, next_observation, done)
        
        def reset(self):
            if hasattr(self.agent, 'reset'):
                self.agent.reset()
    
    return WrappedAgent
