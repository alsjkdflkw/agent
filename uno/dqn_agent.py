"""
Deep Q-Network (DQN) Agent for Uno Game

This module implements a sophisticated learning agent using Deep Q-Learning
with experience replay, target networks, and epsilon-greedy exploration.
"""

import numpy as np
import pickle
from collections import deque
from typing import Any, Dict, List, Tuple, Optional
import gymnasium as gym
from uno.gymnasium_agent import GymnasiumAgent


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    This helps break correlation between consecutive samples and improves
    learning stability.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a random batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(GymnasiumAgent):
    """
    Deep Q-Network agent that learns from experience.
    
    Features:
    - Neural network Q-function approximation (using simple linear model)
    - Experience replay buffer
    - Target network for stable learning
    - Epsilon-greedy exploration
    - Adaptive learning rate
    """
    
    def __init__(self, 
                 action_space: gym.Space,
                 observation_space: gym.Space,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 hidden_size: int = 128):
        """
        Initialize DQN agent.
        
        Args:
            action_space: Environment action space
            observation_space: Environment observation space
            learning_rate: Learning rate for Q-network updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_capacity: Size of replay buffer
            batch_size: Number of samples per training batch
            target_update_freq: Steps between target network updates
            hidden_size: Size of hidden layer in neural network
        """
        super().__init__(action_space, observation_space)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_size = hidden_size
        
        # State representation: flatten (4, 4, 15) observation
        self.state_size = 4 * 4 * 15  # 240
        self.action_size = action_space.n  # 61
        
        # Initialize Q-network (simple linear approximation)
        # In production, you'd use PyTorch or TensorFlow here
        self.q_network = self._init_network()
        self.target_network = self._init_network()
        self._update_target_network()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        self.losses = []
        
    def _init_network(self) -> Dict[str, np.ndarray]:
        """
        Initialize a simple neural network using numpy.
        
        Architecture: Input -> Hidden Layer -> Output
        In production, use PyTorch/TensorFlow for better performance.
        """
        network = {
            'W1': np.random.randn(self.state_size, self.hidden_size) * 0.01,
            'b1': np.zeros((1, self.hidden_size)),
            'W2': np.random.randn(self.hidden_size, self.action_size) * 0.01,
            'b2': np.zeros((1, self.action_size))
        }
        return network
    
    def _forward(self, state: np.ndarray, network: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            state: Flattened state vector
            network: Network weights to use
            
        Returns:
            Q-values for all actions
        """
        # Hidden layer with ReLU activation
        z1 = np.dot(state, network['W1']) + network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Output layer (linear)
        q_values = np.dot(a1, network['W2']) + network['b2']
        
        return q_values
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network."""
        for key in self.q_network.keys():
            self.target_network[key] = self.q_network[key].copy()
    
    def _preprocess_state(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Convert observation to flattened state vector.
        
        Args:
            observation: Game observation dictionary
            
        Returns:
            Flattened numpy array of shape (240,)
        """
        if 'obs' in observation:
            state = observation['obs'].flatten()
        else:
            # Fallback to zeros if obs not available
            state = np.zeros(self.state_size)
        return state
    
    def select_action(self, observation: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current game state
            legal_actions: List of legal action indices
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        
        # Greedy action based on Q-values
        state = self._preprocess_state(observation)
        q_values = self._forward(state.reshape(1, -1), self.q_network).flatten()
        
        # Mask illegal actions
        masked_q = np.full(self.action_size, -np.inf)
        masked_q[legal_actions] = q_values[legal_actions]
        
        return int(np.argmax(masked_q))
    
    def learn(self, 
              observation: Dict[str, Any],
              action: int,
              reward: float,
              next_observation: Dict[str, Any],
              done: bool,
              info: Dict[str, Any]) -> None:
        """
        Learn from a transition using DQN algorithm.
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode ended
            info: Additional information
        """
        # Preprocess states
        state = self._preprocess_state(observation)
        next_state = self._preprocess_state(next_observation)
        
        # Store transition in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Update statistics
        self.steps += 1
        self.total_reward += reward
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch data
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute current Q-values
        q_values = self._forward(states, self.q_network)
        
        # Compute target Q-values using target network
        next_q_values = self._forward(next_states, self.target_network)
        target_q_values = q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + \
                    self.discount_factor * np.max(next_q_values[i])
        
        # Compute loss (MSE)
        loss = np.mean((q_values - target_q_values) ** 2)
        self.losses.append(loss)
        
        # Gradient descent update (simplified)
        # In production, use automatic differentiation
        self._gradient_update(states, target_q_values)
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episodes += 1
    
    def _gradient_update(self, states: np.ndarray, targets: np.ndarray):
        """
        Simplified gradient update using basic backpropagation.
        
        In production, use PyTorch/TensorFlow for automatic differentiation.
        """
        batch_size = states.shape[0]
        
        # Forward pass - save activations
        z1 = np.dot(states, self.q_network['W1']) + self.q_network['b1']
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.q_network['W2']) + self.q_network['b2']
        predictions = z2
        
        # Backward pass
        # Output layer gradient
        dz2 = 2 * (predictions - targets) / batch_size
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        da1 = np.dot(dz2, self.q_network['W2'].T)
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dW1 = np.dot(states.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights with gradient clipping for stability
        max_grad = 1.0
        dW1 = np.clip(dW1, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        
        self.q_network['W1'] -= self.learning_rate * dW1
        self.q_network['b1'] -= self.learning_rate * db1
        self.q_network['W2'] -= self.learning_rate * dW2
        self.q_network['b2'] -= self.learning_rate * db2
    
    def reset(self):
        """Reset agent for new episode."""
        pass
    
    def save(self, filepath: str):
        """
        Save agent to file.
        
        Args:
            filepath: Path to save file
        """
        save_data = {
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'hidden_size': self.hidden_size
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """
        Load agent from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_network = save_data['q_network']
        self.target_network = save_data['target_network']
        self.epsilon = save_data['epsilon']
        self.steps = save_data['steps']
        self.episodes = save_data['episodes']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'total_reward': self.total_reward
        }
